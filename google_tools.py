"""Gmail (read-only) and Google Calendar tools.

One-time sign-in, which opens a browser:
    python google_tools.py
"""

import base64
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE
from tools import Tool, params

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
MAX_BODY_CHARS = 3000
HTML_TAG = re.compile(r"<[^>]+>")
BLANK_LINES = re.compile(r"\n\s*\n+")


def _save(creds: Credentials):
    os.makedirs(os.path.dirname(GOOGLE_TOKEN_FILE), exist_ok=True)
    with open(GOOGLE_TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _credentials(interactive: bool = False) -> Optional[Credentials]:
    creds = None
    if os.path.exists(GOOGLE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save(creds)
        return creds
    if not interactive:
        return None

    flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    _save(creds)
    return creds


def _message_text(payload) -> str:
    """Prefer text/plain parts; fall back to tag-stripped HTML."""
    plain, html = [], []

    def walk(part):
        data = part.get("body", {}).get("data")
        if data:
            text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            mime = part.get("mimeType", "")
            if mime == "text/plain":
                plain.append(text)
            elif mime == "text/html":
                html.append(HTML_TAG.sub(" ", text))
        for sub in part.get("parts", []):
            walk(sub)

    walk(payload)
    text = "\n".join(plain) if plain else "\n".join(html)
    return BLANK_LINES.sub("\n\n", text).strip()[:MAX_BODY_CHARS]


def _local(dt_text: str) -> datetime:
    dt = datetime.fromisoformat(dt_text)
    return dt if dt.tzinfo else dt.astimezone()


def google_tools() -> List[Tool]:
    """Empty until the user has signed in, so AlfreD runs fine without Google."""
    try:
        creds = _credentials()
    except Exception as e:
        logger.warning(f"Google tools disabled: could not refresh sign-in ({e})")
        return []
    if not creds:
        logger.info("Google tools disabled: run `python google_tools.py` to sign in")
        return []

    gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
    calendar = build("calendar", "v3", credentials=creds, cache_discovery=False)

    def gmail_search(query: str = "is:unread in:inbox", max_results: int = 5):
        found = (
            gmail.users()
            .messages()
            .list(userId="me", q=query, maxResults=int(max_results))
            .execute()
            .get("messages", [])
        )
        results = []
        for ref in found:
            msg = (
                gmail.users()
                .messages()
                .get(
                    userId="me",
                    id=ref["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            results.append(
                {
                    "id": msg["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", ""),
                }
            )
        return results or f"No emails match '{query}'."

    def gmail_read(message_id: str):
        msg = (
            gmail.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        return {
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "body": _message_text(msg["payload"]),
        }

    def calendar_events(start: str = "", days: int = 1):
        begin = _local(start) if start else datetime.now().astimezone()
        events = (
            calendar.events()
            .list(
                calendarId="primary",
                timeMin=begin.isoformat(),
                timeMax=(begin + timedelta(days=int(days))).isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=25,
            )
            .execute()
            .get("items", [])
        )
        return [
            {
                "title": e.get("summary", "(no title)"),
                "start": e["start"].get("dateTime", e["start"].get("date")),
                "end": e["end"].get("dateTime", e["end"].get("date")),
                "location": e.get("location", ""),
            }
            for e in events
        ] or "No events in that period."

    def calendar_create_event(
        title: str, start: str, duration_minutes: int = 60, description: str = ""
    ):
        begin = _local(start)
        event = (
            calendar.events()
            .insert(
                calendarId="primary",
                body={
                    "summary": title,
                    "description": description,
                    "start": {"dateTime": begin.isoformat()},
                    "end": {
                        "dateTime": (
                            begin + timedelta(minutes=int(duration_minutes))
                        ).isoformat()
                    },
                },
            )
            .execute()
        )
        return {"created": event.get("summary"), "start": event["start"]["dateTime"]}

    iso = {"type": "string", "description": "ISO 8601 date/time in the user's timezone"}
    return [
        Tool(
            "gmail_search",
            "Search the user's Gmail with Gmail query syntax (e.g. 'is:unread', "
            "'from:alice newer_than:7d'). Returns sender, subject, date, snippet and id.",
            params(query={"type": "string"}, max_results={"type": "integer"}),
            gmail_search,
        ),
        Tool(
            "gmail_read",
            "Read the full text of one email by its id (from gmail_search).",
            params(["message_id"], message_id={"type": "string"}),
            gmail_read,
        ),
        Tool(
            "calendar_events",
            "List events on the user's primary Google Calendar from `start` (default now) "
            "for `days` days.",
            params(start=iso, days={"type": "integer"}),
            calendar_events,
        ),
        Tool(
            "calendar_create_event",
            "Add an event to the user's Google Calendar. Only when the user explicitly "
            "asks to add or schedule something.",
            params(
                ["title", "start"],
                title={"type": "string"},
                start=iso,
                duration_minutes={"type": "integer"},
                description={"type": "string"},
            ),
            calendar_create_event,
        ),
    ]


if __name__ == "__main__":
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        raise SystemExit(
            f"OAuth client file not found: {GOOGLE_CREDENTIALS_FILE}\n"
            "Create a Desktop OAuth client in Google Cloud Console and save its JSON there."
        )
    _credentials(interactive=True)
    print(f"Signed in. Token saved to {GOOGLE_TOKEN_FILE}")
