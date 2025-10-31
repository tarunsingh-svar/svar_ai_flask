import os
import logging
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# ✅ Load environment variables
load_dotenv()

logger = logging.getLogger("AI_Service")

logger.debug(f"🔑 OPENAI_API_KEY Found: {bool(os.getenv('OPENAI_API_KEY'))}")

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_text(prompt):
    logger.debug("🧠 Sending prompt to OpenAI")
    logger.debug(f"Prompt (200): {prompt[:200]}")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()
        logger.debug(f"✅ Generated Result (200): {result[:200]}")
        return result

    except Exception as e:
        logger.error(f"💥 OpenAI Text Generation Error: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error generating"


def transcribe_audio_file(file):
    logger.info("🎤 Transcription started")
    logger.debug(f"File: {file.filename} - {file.mimetype}")

    try:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=(file.filename, file.stream, file.mimetype),
            language="en"
        )

        text = transcript.text.strip()
        logger.debug(f"✅ Transcript (200): {text[:200]}")
        return text

    except Exception as e:
        logger.error(f"💥 OpenAI Transcription Failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
