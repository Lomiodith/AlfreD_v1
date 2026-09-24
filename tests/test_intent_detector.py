"""Worked example — the three pytest mechanics you need, nothing more.

Comments here are teaching notes, not house style. Real tests in this repo
should be as comment-free as the rest of the codebase.
"""

import pytest

from intent_detector import IntentDetector


# A fixture is just a named setup function. Any test that takes `detector` as
# an argument gets a fresh one — pytest matches by parameter name.
@pytest.fixture
def detector():
    return IntentDetector()


def test_exact_prefix_returns_intent_and_query(detector):
    # Plain `assert` is the whole API. pytest rewrites it so a failure shows
    # both sides of the comparison — no assertEqual needed.
    intent, query = detector.detect("search for best laptops")

    assert intent == "search"
    assert query == "best laptops"


# parametrize runs the same test body once per row, reporting each separately.
# This is the workhorse — reach for it before writing a loop inside a test.
@pytest.mark.parametrize(
    "text, expected_query",
    [
        ("search for cats", "cats"),
        ("google cats", "cats"),
        ("look up cats", "cats"),
        ("tell me about cats", "cats"),
    ],
)
def test_query_survives_every_trigger_phrasing(detector, text, expected_query):
    """Regression test: handlers used to re-slice by a hardcoded prefix length,
    so 'tell me about rome' became a search for 'ut rome'."""
    _, query = detector.detect(text)

    assert query == expected_query


def test_unmatched_text_falls_through_to_general(detector):
    intent, query = detector.detect("the sky looks orange today")

    assert intent == "general"
    assert query == "the sky looks orange today"


@pytest.mark.parametrize(
    "text",
    [
        ("that's quite nice"),
        ("a quitter never wins"),
        ("exits are locked"),
        ("I was researching that"),
        ("the commander said"),
        ("be helpful"),
    ],
)
def test_trigger_inside_a_word_does_not_route(detector, text):
    intent, _ = detector.detect(text)

    assert intent == "general"


@pytest.mark.parametrize(
    "text, expected_query",
    [
        ("Read file notes.txt.", "notes.txt"),
        ("Run command, ls -la.", "ls -la"),
        ("Scrape https://example.com for links.", "https://example.com for links"),
        ("Search for C++?", "C++"),
    ],
)
def test_transcript_punctuation_is_trimmed_from_query(detector, text, expected_query):
    _, query = detector.detect(text)

    assert query == expected_query
