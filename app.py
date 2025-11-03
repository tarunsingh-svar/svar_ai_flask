from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.ai_service import generate_text, transcribe_audio_file
import os
import logging
import traceback
import time

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FlaskApp")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "✅ Flask AI API Running!"


@app.route("/summarize", methods=["GET"])
def summarize_text():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Missing text param"}), 400

    result = generate_text(f"Summarize clearly:\n{text}")
    return jsonify({"summary": result})


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    transcript = transcribe_audio_file(request.files["file"])
    return jsonify({"transcript": transcript})


def gen(title, prompt):
    logger.info(f"✨ {title} API")
    result = generate_text(prompt)
    return result


# =============== SOCIAL / CREATOR =================
@app.route("/generate_x_post", methods=["POST"])
def generate_x_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("X Post", f"Engaging X Post:\n{text}")})


@app.route("/generate_x_thread", methods=["POST"])
def generate_x_thread():
    text = request.json.get("text", "")
    return jsonify({"result": gen("X Thread", f"Write short threaded tweets:\n{text}")})


@app.route("/generate_facebook_post", methods=["POST"])
def generate_facebook_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("FB Post", f"Friendly Facebook post:\n{text}")})


@app.route("/generate_linkedin_post", methods=["POST"])
def generate_linkedin_post():
    text = request.json.get("text", "")
    return jsonify({"result": gen("LinkedIn Post", f"Professional LinkedIn post:\n{text}")})


@app.route("/generate_video_script", methods=["POST"])
def generate_video_script():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Reel/TikTok Script", f"Attention grabbing video script:\n{text}")})


@app.route("/generate_content_outline", methods=["POST"])
def generate_content_outline():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Content Outline", f"Structured outline:\n{text}")})


# =============== PRODUCTIVITY =================
@app.route("/generate_quick_list", methods=["POST"])
def generate_quick_list():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Quick List", f"Convert into short bullet list:\n{text}")})


@app.route("/generate_todo_list", methods=["POST"])
def generate_todo_list():
    text = request.json.get("text", "")
    return jsonify({"result": gen("To-do List", f"Actionable tasks list:\n{text}")})


@app.route("/generate_meeting_notes", methods=["POST"])
def generate_meeting_notes():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Meeting Notes", f"Clean meeting notes:\n{text}")})


# =============== WORK COLLAB =================
@app.route("/generate_daily_standup", methods=["POST"])
def generate_daily_standup():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Daily Standup", f"Classic daily standup format:\n{text}")})


@app.route("/generate_feature_discussion", methods=["POST"])
def generate_feature_discussion():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Feature Discussion", f"Team feature discussion format:\n{text}")})


@app.route("/generate_interview_summary", methods=["POST"])
def generate_interview_summary():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Interview Summary", f"Insights from user interview:\n{text}")})


@app.route("/generate_delegation_note", methods=["POST"])
def generate_delegation_note():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Delegation Note", f"Delegate tasks clearly:\n{text}")})


# =============== EMAILS =================
@app.route("/generate_email_casual", methods=["POST"])
def generate_email_casual():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Casual Email", f"Friendly short email:\n{text}")})


@app.route("/generate_email_formal", methods=["POST"])
def generate_email_formal():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Formal Email", f"Structured professional email:\n{text}")})


# =============== LEARNING =================
@app.route("/generate_lecture_summary", methods=["POST"])
def generate_lecture_summary():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Lecture Summary", f"Clear class/lecture summary:\n{text}")})


# =============== JOURNALING =================
@app.route("/generate_journal", methods=["POST"])
def generate_journal():
    text = request.json.get("text", "")
    return jsonify({"result": gen("Journal Entry", f"Short personal journal entry:\n{text}")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
