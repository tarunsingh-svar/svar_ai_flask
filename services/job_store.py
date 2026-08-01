"""Transcription job state, stored in Supabase.

Job state used to live in a module-level dict, which silently broke under
`gunicorn --workers 2`: the worker that started a job was often not the worker
that answered the status poll, so clients saw a 404 partway through.

Reads and writes go through PostgREST with the service role key. Every read is
scoped to a user id supplied by the caller, so one user cannot poll another
user's job.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("JobStore")

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

_TABLE = "transcription_jobs"
_TIMEOUT = 15.0


class JobStoreError(RuntimeError):
    pass


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    if not SUPABASE_URL or not SERVICE_ROLE_KEY:
        raise JobStoreError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set."
        )
    headers = {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _endpoint() -> str:
    return f"{SUPABASE_URL}/rest/v1/{_TABLE}"


def create_job(user_id: str) -> str:
    response = httpx.post(
        _endpoint(),
        headers=_headers({"Prefer": "return=representation"}),
        json={"user_id": user_id, "status": "pending"},
        timeout=_TIMEOUT,
    )
    if response.status_code not in (200, 201):
        logger.error(f"create_job failed {response.status_code}: {response.text}")
        raise JobStoreError("Could not create the transcription job.")

    rows = response.json()
    if not rows:
        raise JobStoreError("Could not create the transcription job.")
    return rows[0]["id"]


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    response = httpx.patch(
        f"{_endpoint()}?id=eq.{job_id}",
        headers=_headers(),
        json=fields,
        timeout=_TIMEOUT,
    )
    if response.status_code not in (200, 204):
        # A failed status write must not kill the worker thread mid-job.
        logger.error(f"update_job failed {response.status_code}: {response.text}")


def get_job(job_id: str, user_id: str) -> dict | None:
    response = httpx.get(
        f"{_endpoint()}?id=eq.{job_id}&user_id=eq.{user_id}"
        "&select=status,transcript,error",
        headers=_headers(),
        timeout=_TIMEOUT,
    )
    if response.status_code != 200:
        logger.error(f"get_job failed {response.status_code}: {response.text}")
        raise JobStoreError("Could not read the transcription job.")

    rows = response.json()
    return rows[0] if rows else None
