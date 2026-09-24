import pytest

from intent_detector import IntentDetector


@pytest.fixture
def detector():
    return IntentDetector()


@pytest.mark.parametrize(
    "text, intent",
    [
        ("exit", "terminate"),
        ("Exit.", "terminate"),
        ("Okay, shut down please.", "terminate"),
        ("Goodbye!", "terminate"),
        ("Clear context.", "clear_context"),
        ("Start fresh", "clear_context"),
        ("Show commands.", "show_commands"),
        ("Performance stats", "performance"),
        ("shut dawn", "terminate"),
        ("Text mode.", "voice_off"),
        ("No.", "decline"),
        ("No, thank you.", "decline"),
        ("Nope", "decline"),
        ("That's all, thanks.", "decline"),
        ("Voice mode, please", "voice_on"),
    ],
)
def test_control_command_as_whole_utterance(detector, text, intent):
    assert detector.detect(text) == intent


@pytest.mark.parametrize(
    "text",
    [
        "How do I exit vim?",
        "Can you quit smoking easily?",
        "Quit smoking tips",
        "What's the performance of the S&P 500?",
        "I need help with my essay",
        "Search for the best laptops",
        "a quitter never wins",
        "exits are locked",
        "the commander said",
        "No, I meant the other city",
        "Stop the pasta timer",
    ],
)
def test_sentences_containing_triggers_go_to_the_llm(detector, text):
    assert detector.detect(text) == "general"
