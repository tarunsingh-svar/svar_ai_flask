from dataclasses import dataclass

SHARED_SYSTEM_PROMPT = """You rewrite voice-note transcripts into structured documents.
Rules:
- Use ONLY facts stated in the transcript. Do not invent names, dates, or decisions.
- If information is missing for a section, omit that section — never guess.
- Preserve speaker names and numbers exactly when mentioned.
- Output ONLY the final document. No intro sentence, no "Here is your...".
- The transcript may be in English or Hinglish. Output in English."""


@dataclass
class RewriteConfig:
    rewrite_id: str
    task_instruction: str
    output_template: str
    max_output_tokens: int
    temperature: float


REWRITE_CONFIGS: dict[str, RewriteConfig] = {

    # ============================================================
    # CORE PRODUCTIVITY
    # ============================================================

    "quick_list": RewriteConfig(
        rewrite_id="quick_list",
        task_instruction=(
            "Convert the transcript into a concise bullet-point list of the main thoughts, "
            "facts, or items mentioned. Maximum 10 bullets."
        ),
        output_template=(
            "Use plain bullet points (- item). "
            "Each bullet is a single line. No sub-bullets. No section headings."
        ),
        max_output_tokens=400,
        temperature=0.2,
    ),

    "meeting_notes": RewriteConfig(
        rewrite_id="meeting_notes",
        task_instruction=(
            "Transform the transcript into clean, structured meeting notes."
        ),
        output_template=(
            "## Key Discussion\n"
            "- <point>\n\n"
            "## Decisions\n"
            "- <decision>\n\n"
            "## Action Items\n"
            "- [ ] <Owner> — <Task> — <Due date if mentioned>"
        ),
        max_output_tokens=900,
        temperature=0.2,
    ),

    "todo_list": RewriteConfig(
        rewrite_id="todo_list",
        task_instruction=(
            "Extract all action items and tasks from the transcript into a checklist."
        ),
        output_template=(
            "Use the format: - [ ] <task>\n"
            "Each line is one actionable task. Group by person if multiple people are mentioned."
        ),
        max_output_tokens=500,
        temperature=0.2,
    ),

    # ============================================================
    # WORK MEETINGS & COLLABORATION
    # ============================================================

    "daily_standup": RewriteConfig(
        rewrite_id="daily_standup",
        task_instruction=(
            "Reformat the transcript as a classic daily standup update."
        ),
        output_template=(
            "**Yesterday:**\n- <what was completed>\n\n"
            "**Today:**\n- <what is planned>\n\n"
            "**Blockers:**\n- <any blockers, or 'None' if none mentioned>"
        ),
        max_output_tokens=500,
        temperature=0.2,
    ),

    "feature_discussion": RewriteConfig(
        rewrite_id="feature_discussion",
        task_instruction=(
            "Summarise the transcript as structured product or feature discussion notes."
        ),
        output_template=(
            "## Summary\n<one-paragraph summary>\n\n"
            "## Key Points\n- <point>\n\n"
            "## Open Questions\n- <question>\n\n"
            "## Next Steps\n- <step>"
        ),
        max_output_tokens=800,
        temperature=0.3,
    ),

    "interview_summary": RewriteConfig(
        rewrite_id="interview_summary",
        task_instruction=(
            "Summarise the transcript as a user interview summary, extracting insights and notable quotes."
        ),
        output_template=(
            "## Key Insights\n- <insight>\n\n"
            "## Notable Quotes\n- \"<exact quote from transcript>\"\n\n"
            "## Follow-ups\n- <follow-up question or action>"
        ),
        max_output_tokens=800,
        temperature=0.3,
    ),

    "delegation_note": RewriteConfig(
        rewrite_id="delegation_note",
        task_instruction=(
            "Rewrite the transcript as a clear delegation note using Who / What / When structure."
        ),
        output_template=(
            "**Who:** <name of person being delegated to>\n"
            "**What:** <task or responsibility being delegated>\n"
            "**When:** <deadline or timeframe if mentioned>\n"
            "**Notes:** <any additional context or dependencies>"
        ),
        max_output_tokens=500,
        temperature=0.2,
    ),

    # ============================================================
    # PROFESSIONAL WRITING
    # ============================================================

    "email_casual": RewriteConfig(
        rewrite_id="email_casual",
        task_instruction=(
            "Turn the transcript into a short, friendly casual email."
        ),
        output_template=(
            "Subject: <subject line>\n\n"
            "Hi <name if mentioned, else omit greeting>,\n\n"
            "<body — friendly, conversational tone, 80–150 words>\n\n"
            "<sign-off>"
        ),
        max_output_tokens=400,
        temperature=0.4,
    ),

    "email_formal": RewriteConfig(
        rewrite_id="email_formal",
        task_instruction=(
            "Turn the transcript into a structured, professional formal email."
        ),
        output_template=(
            "Subject: <subject line>\n\n"
            "Dear <name if mentioned, else 'Sir/Madam'>,\n\n"
            "<body — formal tone, clear paragraphs, 100–200 words>\n\n"
            "Regards,\n<sign-off>"
        ),
        max_output_tokens=500,
        temperature=0.3,
    ),

    # ============================================================
    # CREATOR
    # ============================================================

    "x_post": RewriteConfig(
        rewrite_id="x_post",
        task_instruction=(
            "Write a single engaging X (Twitter) post based on the transcript."
        ),
        output_template=(
            "One tweet, maximum 280 characters. "
            "Hook the reader in the first line. "
            "At most 2 relevant hashtags. No filler phrases."
        ),
        max_output_tokens=150,
        temperature=0.6,
    ),

    "x_thread": RewriteConfig(
        rewrite_id="x_thread",
        task_instruction=(
            "Turn the transcript into an engaging X (Twitter) thread."
        ),
        output_template=(
            "Number each tweet: 1/, 2/, 3/ ...\n"
            "Each tweet is ≤280 characters.\n"
            "First tweet is the hook. Last tweet is the takeaway or CTA.\n"
            "Aim for 4–8 tweets based on content depth."
        ),
        max_output_tokens=700,
        temperature=0.5,
    ),

    "linkedin_post": RewriteConfig(
        rewrite_id="linkedin_post",
        task_instruction=(
            "Write a professional LinkedIn post based on the transcript."
        ),
        output_template=(
            "Opening line that hooks the reader.\n\n"
            "2–4 short paragraphs (150–300 words total).\n\n"
            "Closing thought or question to drive engagement.\n\n"
            "At most 3 relevant hashtags on a separate line at the end."
        ),
        max_output_tokens=400,
        temperature=0.5,
    ),

    "short_video_script": RewriteConfig(
        rewrite_id="short_video_script",
        task_instruction=(
            "Write an attention-catching short video script (Reel / TikTok / YouTube Short) "
            "based on the transcript."
        ),
        output_template=(
            "**Hook (0–3s):** <opening line spoken to camera>\n\n"
            "**Main Points:**\n- <point 1>\n- <point 2>\n- <point 3 if needed>\n\n"
            "**CTA (last 3s):** <call to action>"
        ),
        max_output_tokens=600,
        temperature=0.5,
    ),

    "content_outline": RewriteConfig(
        rewrite_id="content_outline",
        task_instruction=(
            "Create a structured content outline from the transcript, "
            "suitable for a post, video, or newsletter."
        ),
        output_template=(
            "# <Title>\n\n"
            "## <Section 1>\n- <sub-point>\n\n"
            "## <Section 2>\n- <sub-point>\n\n"
            "## <Section 3 if needed>\n- <sub-point>"
        ),
        max_output_tokens=700,
        temperature=0.4,
    ),

    # ============================================================
    # LEARNING & RESEARCH
    # ============================================================

    "lecture_summary": RewriteConfig(
        rewrite_id="lecture_summary",
        task_instruction=(
            "Summarise the transcript as clean lecture or class notes."
        ),
        output_template=(
            "## Key Concepts\n- <concept>\n\n"
            "## Details & Examples\n- <detail or example>\n\n"
            "## Takeaways\n- <main takeaway>"
        ),
        max_output_tokens=800,
        temperature=0.2,
    ),

    # ============================================================
    # JOURNALING & PERSONAL
    # ============================================================

    "journal": RewriteConfig(
        rewrite_id="journal",
        task_instruction=(
            "Rewrite the transcript as a personal journal entry in first-person voice."
        ),
        output_template=(
            "A flowing, reflective first-person journal entry. "
            "100–200 words. Preserve the speaker's emotional tone. "
            "No headings, no bullet points — just natural prose."
        ),
        max_output_tokens=400,
        temperature=0.5,
    ),
}
