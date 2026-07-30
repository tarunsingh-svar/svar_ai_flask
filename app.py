from flask import Flask, g, request, jsonify
from dotenv import load_dotenv
from services.ai_service import (
    generate_text,
    generate_rewrite,
    create_transcription_job,
    create_transcription_job_from_url,
    get_transcription_job,
)
from services.auth import authenticate_request
from services.rewrite_registry import REWRITE_CONFIGS
import logging
import traceback

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FlaskApp")

app = Flask(__name__)

# Routes reachable without a Supabase session. Everything else is authenticated
# by the before_request hook below, so a new endpoint is private by default.
PUBLIC_ENDPOINTS = {"home"}


@app.before_request
def enforce_authentication():
    if request.method == "OPTIONS":
        return None
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    return authenticate_request()


@app.route("/", methods=["GET"])
def home():
    """Unauthenticated health check. Clients ping this to wake the dyno."""
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
    """Multipart upload path, used by the mobile app."""
    if "file" not in request.files:
        return jsonify({"error": "file missing"}), 400

    try:
        job_id = create_transcription_job(g.user_id, request.files["file"])
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    except Exception as e:
        logger.error(f"Failed to start transcription job: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Could not start transcription."}), 500


@app.route("/transcribe_url", methods=["POST"])
def transcribe_audio_url():
    """Storage-object path, used by the web app.

    The browser uploads directly to Supabase Storage and sends a signed URL, so
    the audio never passes through a serverless request body.
    """
    body = request.get_json(silent=True) or {}
    audio_url = (body.get("audio_url") or "").strip()

    if not audio_url.startswith("https://"):
        return jsonify({"error": "audio_url missing or not https"}), 400

    try:
        job_id = create_transcription_job_from_url(g.user_id, audio_url)
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    except Exception as e:
        logger.error(f"Failed to start transcription job from url: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Could not start transcription."}), 500


@app.route("/transcribe/status/<job_id>", methods=["GET"])
def transcribe_status(job_id):
    try:
        job = get_transcription_job(job_id, g.user_id)
    except Exception as e:
        logger.error(f"Failed to read transcription job {job_id}: {e}")
        return jsonify({"error": "Could not read the job status."}), 503

    # Also covers jobs belonging to another user: the lookup is scoped by
    # user_id, so someone else's job id is indistinguishable from a missing one.
    if job is None:
        return jsonify({"error": "job not found"}), 404

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
