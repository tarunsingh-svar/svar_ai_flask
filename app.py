from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.ai_service import (
    generate_text,
    generate_rewrite,
    transcribe_audio_file,
    create_transcription_job,
    get_transcription_job,
    clear_transcription_job,
)
from services.rewrite_registry import REWRITE_CONFIGS
import os
import logging
import traceback

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


@app.route("/summarize", methods=["GET", "POST"])
def summarize_text():
    text = None
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        text = body.get("text")
    else:
        text = request.args.get("text")

    if not text:
        return jsonify({"error": "Missing text param"}), 400

    result = generate_text(f"Summarize clearly:\n{text}")
    return jsonify({"summary": result})


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    try:
        job_id = create_transcription_job(request.files["file"])
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    except Exception as e:
        logger.error(f"Failed to start transcription job: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe/status/<job_id>", methods=["GET"])
def transcribe_status(job_id):
    job = get_transcription_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404

    if job.get("status") in ("complete", "failed"):
        clear_transcription_job(job_id)

    return jsonify(job)


# ============================================================
# Shared helper — all rewrite routes use this
# ============================================================

def _rewrite(rewrite_id: str):
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    config = REWRITE_CONFIGS[rewrite_id]
    logger.info(f"✨ Rewrite [{rewrite_id}] input_len={len(text)}")
    return jsonify({"result": generate_rewrite(config, text)})


# ============================================================
# SOCIAL / CREATOR
# ============================================================

@app.route("/generate_x_post", methods=["POST"])
def generate_x_post():
    return _rewrite("x_post")


@app.route("/generate_x_thread", methods=["POST"])
def generate_x_thread():
    return _rewrite("x_thread")


@app.route("/generate_facebook_post", methods=["POST"])
def generate_facebook_post():
    # Facebook post re-uses the LinkedIn config as a reasonable default;
    # swap to its own config if a dedicated facebook entry is added later.
    return _rewrite("linkedin_post")


@app.route("/generate_linkedin_post", methods=["POST"])
def generate_linkedin_post():
    return _rewrite("linkedin_post")


@app.route("/generate_video_script", methods=["POST"])
def generate_video_script():
    return _rewrite("short_video_script")


@app.route("/generate_content_outline", methods=["POST"])
def generate_content_outline():
    return _rewrite("content_outline")


# ============================================================
# PRODUCTIVITY
# ============================================================

@app.route("/generate_quick_list", methods=["POST"])
def generate_quick_list():
    return _rewrite("quick_list")


@app.route("/generate_todo_list", methods=["POST"])
def generate_todo_list():
    return _rewrite("todo_list")


@app.route("/generate_meeting_notes", methods=["POST"])
def generate_meeting_notes():
    return _rewrite("meeting_notes")


# ============================================================
# WORK COLLAB
# ============================================================

@app.route("/generate_daily_standup", methods=["POST"])
def generate_daily_standup():
    return _rewrite("daily_standup")


@app.route("/generate_feature_discussion", methods=["POST"])
def generate_feature_discussion():
    return _rewrite("feature_discussion")


@app.route("/generate_interview_summary", methods=["POST"])
def generate_interview_summary():
    return _rewrite("interview_summary")


@app.route("/generate_delegation_note", methods=["POST"])
def generate_delegation_note():
    return _rewrite("delegation_note")


# ============================================================
# EMAILS
# ============================================================

@app.route("/generate_email_casual", methods=["POST"])
def generate_email_casual():
    return _rewrite("email_casual")


@app.route("/generate_email_formal", methods=["POST"])
def generate_email_formal():
    return _rewrite("email_formal")


# ============================================================
# LEARNING
# ============================================================

@app.route("/generate_lecture_summary", methods=["POST"])
def generate_lecture_summary():
    return _rewrite("lecture_summary")


# ============================================================
# JOURNALING
# ============================================================

@app.route("/generate_journal", methods=["POST"])
def generate_journal():
    return _rewrite("journal")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
