import base64

from google_tools import _local, _message_text


def _part(mime, text):
    data = base64.urlsafe_b64encode(text.encode()).decode()
    return {"mimeType": mime, "body": {"data": data}}


def test_prefers_plain_text_part():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            _part("text/plain", "Hello there"),
            _part("text/html", "<b>Hello</b>"),
        ],
    }

    assert _message_text(payload) == "Hello there"


def test_falls_back_to_stripped_html():
    payload = {"mimeType": "text/html", **_part("text/html", "<p>Hi <b>you</b></p>")}

    assert _message_text(payload) == "Hi  you"


def test_naive_times_get_the_local_timezone():
    assert _local("2026-09-25T15:00").tzinfo is not None
    assert _local("2026-09-25T15:00+00:00").utcoffset().total_seconds() == 0
