from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.ai_service import generate_text, transcribe_audio_file
import os
import logging
import traceback
import time

# ✅ Load env file
load_dotenv()

# ✅ Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FlaskApp")

logger.debug("🚀 Flask starting...")
logger.debug(f"🔑 OPENAI Key Found: {bool(os.getenv('OPENAI_API_KEY'))}")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    logger.info("🏠 Home endpoint hit")
    return "✅ Flask AI API Running!"


# ✅ Summarize API
@app.route("/summarize", methods=["GET"])
def summarize_text():
    text = request.args.get("text")
    logger.info("📝 /summarize endpoint hit")

    if not text:
        logger.error("❌ Missing `text` param")
        return jsonify({"error": "Missing text param"}), 400

    logger.debug(f"📩 Summarize Input: {text[:200]}")
    prompt = f"Summarize clearly:\n{text.strip()}"

    try:
        result = generate_text(prompt)
        logger.debug(f"📤 Summary Output: {result[:200]}")
        return jsonify({"summary": result})

    except Exception as e:
        logger.error(f"💥 Summarize failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ✅ Transcribe API
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    start = time.time()
    logger.info("🎧 /transcribe endpoint hit")

    if "file" not in request.files:
        logger.error("❌ file missing")
        return jsonify({"error": "file missing"}), 400

    file = request.files["file"]
    logger.debug(f"📎 File Received: {file.filename}, MIME: {file.mimetype}")

    try:
        transcript = transcribe_audio_file(file)
        elapsed = round(time.time() - start, 2)

        logger.info(f"✅ Transcription done in {elapsed}s")
        logger.debug(f"📤 Transcript: {transcript[:200]}")

        return jsonify({"transcript": transcript})

    except Exception as e:
        logger.error(f"💥 Transcription failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ✅ Text generation helper endpoints
def gen(prompt_title, text):
    logger.info(f"✨ {prompt_title} generation")
    logger.debug(f"Input: {text[:150]}")
    return generate_text(text)


@app.route("/generate_x_post", methods=["POST"])
def generate_x_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("X Post", f"Write an engaging X post:\n{text}")})


@app.route("/generate_x_thread", methods=["POST"])
def generate_x_thread():
    text = request.json.get("text", "")
    return jsonify({"result": gen("X Thread", f"Make a short X thread in points:\n{text}")})


@app.route("/generate_facebook_post", methods=["POST"])
def generate_facebook_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Facebook Post", f"Friendly Facebook post:\n{text}")})


@app.route("/generate_linkedin_post", methods=["POST"])
def generate_linkedin_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("LinkedIn Post", f"Professional LinkedIn post:\n{text}")})


@app.route("/generate_meeting_notes", methods=["POST"])
def generate_meeting_notes():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Meeting Notes", f"Proper meeting notes:\n{text}")})


@app.route("/generate_journal", methods=["POST"])
def generate_journal():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Journal", f"Journal entry:\n{text}")})


if __name__ == "__main__":
    logger.info("🚀 Flask server running on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
