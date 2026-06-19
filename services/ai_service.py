import os
import json
import glob
import shutil
import logging
import traceback
import threading
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from sarvamai import SarvamAI

load_dotenv()
logger = logging.getLogger("AI_Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

_openai_client = None
_sarvam_client = None
_transcription_jobs = {}
_transcription_jobs_lock = threading.Lock()


def create_transcription_job(file_storage):
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_jobs", job_id)
    os.makedirs(temp_dir, exist_ok=True)
    filename = file_storage.filename or "audio.m4a"
    temp_audio_path = os.path.join(temp_dir, filename)
    file_storage.save(temp_audio_path)

    with _transcription_jobs_lock:
        _transcription_jobs[job_id] = {
            "status": "pending",
            "transcript": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_transcription_job,
        args=(job_id, temp_audio_path, temp_dir),
        daemon=True,
    )
    thread.start()
    return job_id


def get_transcription_job(job_id):
    with _transcription_jobs_lock:
        job = _transcription_jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


def clear_transcription_job(job_id):
    with _transcription_jobs_lock:
        _transcription_jobs.pop(job_id, None)


def _run_transcription_job(job_id, temp_audio_path, temp_dir):
    try:
        with _transcription_jobs_lock:
            if job_id in _transcription_jobs:
                _transcription_jobs[job_id]["status"] = "processing"

        transcript = transcribe_audio_path(temp_audio_path)

        with _transcription_jobs_lock:
            if job_id in _transcription_jobs:
                _transcription_jobs[job_id]["status"] = "complete"
                _transcription_jobs[job_id]["transcript"] = transcript
    except Exception as e:
        logger.error(f"Transcription job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        with _transcription_jobs_lock:
            if job_id in _transcription_jobs:
                _transcription_jobs[job_id]["status"] = "failed"
                _transcription_jobs[job_id]["error"] = str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_sarvam_client():
    global _sarvam_client
    if _sarvam_client is None:
        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY is not set in .env")
        _sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _sarvam_client


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
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"generate_rewrite error [{config.rewrite_id}]: {e}")
        return "Error generating rewrite"


def transcribe_audio_file(file):
    temp_dir = "temp"
    out_dir = os.path.join(temp_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_dir, file.filename or "audio.m4a")
    with open(temp_audio_path, "wb") as f:
        shutil.copyfileobj(file.stream, f)
    try:
        return transcribe_audio_path(temp_audio_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def transcribe_audio_path(temp_audio_path):
    logger.info("🎤 Sarvam transcription with English output start")

    temp_dir = os.path.dirname(temp_audio_path)
    out_dir = os.path.join(temp_dir, "out")

    try:
        os.makedirs(out_dir, exist_ok=True)

        job = _get_sarvam_client().speech_to_text_job.create_job(
            model="saarika:v2.5",
            with_diarization=True
        )

        job.upload_files([temp_audio_path])
        job.start()
        job.wait_until_complete()

        if job.is_failed():
            return "Sarvam job failed"

        job.download_outputs(out_dir)

        json_files = glob.glob(os.path.join(out_dir, "*.json"))
        if not json_files:
            return "No transcript file from Sarvam"

        with open(json_files[0], "r", encoding="utf-8") as jf:
            res = json.load(jf)

        entries = res.get("diarized_transcript", {}).get("entries", [])
        if not entries:
            # fallback if diarization not available
            txt = res.get("transcript", "")
            return txt.strip()

        formatted_lines = []
        for e in entries:
            speaker = e.get("speaker_id", "Speaker").replace("_", " ").title()

            # 1️⃣ Try English transliteration (Hinglish)
            english_text = (
                e.get("transcription_output", {})
                .get("transcriptions", {})
                .get("hi-IN", {})
                .get("transliteration", {})
                .get("en", {})
                .get("text", "")
            )

            # 2️⃣ Try if there’s already an English transcription
            if not english_text:
                english_text = (
                    e.get("transcription_output", {})
                    .get("transcriptions", {})
                    .get("en-IN", {})
                    .get("text", "")
                )

            # 3️⃣ Fallback to original transcript if all else fails
            if not english_text:
                english_text = e.get("transcript", "")

            formatted_lines.append(f"Speaker {speaker}: {english_text}")

        final_text = "\n".join(formatted_lines).strip()
        return final_text

    except Exception as e:
        logger.error(f"Sarvam Error: {e}")
        logger.error(traceback.format_exc())
        return "Transcription error"

    finally:
        out_only = os.path.join(os.path.dirname(temp_audio_path), "out")
        if os.path.exists(out_only):
            shutil.rmtree(out_only, ignore_errors=True)