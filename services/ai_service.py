import os
import json
import glob
import shutil
import logging
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from sarvamai import SarvamAI

load_dotenv()
logger = logging.getLogger("AI_Service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)


def generate_text(prompt):
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return "Error generating text"


def transcribe_audio_file(file):
    logger.info("🎤 Sarvam transcription with Hinglish start")

    temp_dir = "temp"
    out_dir = os.path.join(temp_dir, "out")

    try:
        os.makedirs(out_dir, exist_ok=True)

        temp_audio_path = os.path.join(temp_dir, file.filename or "audio.m4a")
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(file.stream, f)

        job = sarvam_client.speech_to_text_job.create_job(
            language_code="hi-IN",
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
            txt = res.get("transcript", "")
            return txt.strip()

        formatted_lines = []
        for e in entries:
            speaker = e.get("speaker_id", "Speaker").replace("_"," ").title()

            # ✅ Hinglish first (en transliteration)
            hinglish = (
                e.get("transcription_output", {})
                .get("transcriptions", {})
                .get("hi-IN", {})
                .get("transliterations", {})
                .get("en", {})
                .get("text", "")
            )

            # fallback to raw transcript if Hinglish missing
            if not hinglish:
                hinglish = e.get("transcript", "")

            formatted_lines.append(f"Speaker {speaker}: {hinglish}")

        final_text = "\n".join(formatted_lines).strip()
        return final_text


    except Exception as e:
        logger.error(f"Sarvam Error: {e}")
        logger.error(traceback.format_exc())
        return "Transcription error"

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
