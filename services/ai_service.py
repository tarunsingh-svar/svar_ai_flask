import os
import shutil
import logging
import traceback
import threading
import httpx
from dotenv import load_dotenv
from openai import OpenAI

from services import stt
from services.job_store import create_job, get_job, update_job

# Re-exported: this module was the historical home of the error type.
from services.stt import TranscriptionError  # noqa: F401

load_dotenv()
logger = logging.getLogger("AI_Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Guards against a malicious or accidental multi-gigabyte download.
MAX_AUDIO_BYTES = 500 * 1024 * 1024

_openai_client = None


def _job_dir(job_id):
    path = os.path.join("temp_jobs", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def create_transcription_job(user_id, file_storage, language=None, locale=None):
    """Starts a job from a direct multipart upload (used by the mobile app).

    [language] is the user's Settings choice (None or "auto" to detect) and
    [locale] is the device locale; together they pick the STT provider.
    """
    job_id = create_job(user_id, requested_language=language)
    temp_dir = _job_dir(job_id)
    filename = os.path.basename(file_storage.filename or "audio.m4a")
    temp_audio_path = os.path.join(temp_dir, filename)
    file_storage.save(temp_audio_path)

    _start_worker(job_id, temp_audio_path, temp_dir, language, locale)
    return job_id


def create_transcription_job_from_url(user_id, audio_url, language=None, locale=None):
    """Starts a job from a Storage object (used by the web app).

    The browser uploads straight to Supabase Storage, which sidesteps the 4.5 MB
    serverless request body limit that a proxied upload would hit.
    """
    job_id = create_job(user_id, requested_language=language)
    temp_dir = _job_dir(job_id)

    try:
        temp_audio_path = _download_audio(audio_url, temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        update_job(job_id, status="failed", error=str(e))
        raise

    _start_worker(job_id, temp_audio_path, temp_dir, language, locale)
    return job_id


def _download_audio(audio_url, temp_dir):
    extension = os.path.splitext(audio_url.split("?")[0])[1] or ".m4a"
    destination = os.path.join(temp_dir, f"audio{extension}")

    written = 0
    with httpx.stream("GET", audio_url, timeout=120.0, follow_redirects=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 256):
                written += len(chunk)
                if written > MAX_AUDIO_BYTES:
                    raise ValueError("That recording is too large to transcribe.")
                handle.write(chunk)

    if written == 0:
        raise ValueError("The recording was empty.")

    return destination


def _start_worker(job_id, temp_audio_path, temp_dir, language=None, locale=None):
    thread = threading.Thread(
        target=_run_transcription_job,
        args=(job_id, temp_audio_path, temp_dir, language, locale),
        daemon=True,
    )
    thread.start()


def get_transcription_job(job_id, user_id):
    return get_job(job_id, user_id)


def _run_transcription_job(job_id, temp_audio_path, temp_dir, language=None, locale=None):
    try:
        update_job(job_id, status="processing")
        result = stt.transcribe(temp_audio_path, language=language, locale=locale)
        # Recording which provider ran is the difference between diagnosing a bad
        # transcript and guessing at it.
        update_job(
            job_id,
            status="complete",
            transcript=result.to_text(),
            provider=result.provider,
        )
    except Exception as e:
        logger.error(f"Transcription job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        update_job(job_id, status="failed", error=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def generate_text(prompt):
    try:
        resp = _get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return "Error generating text"


def generate_rewrite(config, transcript: str) -> str:
    """Generate a structured rewrite using the given RewriteConfig.

    The transcript is passed in full to support long recordings — GPT-5.4-nano
    has a 400K context window so there is no practical ceiling. A 24K-character
    soft cap (~6K tokens) is applied as a guard against unexpectedly oversized
    payloads (e.g. multi-hour diarized sessions) that would slow the Render
    cold-start path. Raise or remove this cap per-option when needed.
    """
    from services.rewrite_registry import SHARED_SYSTEM_PROMPT

    if not transcript or not transcript.strip():
        return "No transcript available to rewrite."

    MAX_INPUT_CHARS = 24_000
    trimmed = transcript.strip()
    if len(trimmed) > MAX_INPUT_CHARS:
        logger.warning(
            f"generate_rewrite: transcript trimmed from {len(trimmed)} to "
            f"{MAX_INPUT_CHARS} chars for rewrite_id={config.rewrite_id}"
        )
        trimmed = trimmed[:MAX_INPUT_CHARS]

    user_message = (
        f"Task: {config.task_instruction}\n\n"
        f"Output format:\n{config.output_template}\n\n"
        f"Transcript:\n---\n{trimmed}\n---"
    )

    try:
        resp = _get_openai_client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"generate_rewrite error [{config.rewrite_id}]: {e}")
        return "Error generating rewrite"


def transcribe_audio_path(temp_audio_path, language=None, locale=None):
    """Transcribes a local file synchronously. Prefer the job API for requests.

    Raises TranscriptionError rather than returning a sentinel string: callers
    record job state, and a returned message would be saved as the transcript.
    """
    return stt.transcribe(temp_audio_path, language=language, locale=locale).to_text()