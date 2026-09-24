import pytest

from language_guard import spoken_language, unexpected_language

ALLOWED = ("en", "ro")


@pytest.mark.parametrize(
    "text, language",
    [
        ("Ku mnie wymá fary klucz.", "Polish"),
        ("Альфред.", "Ukrainian"),
    ],
)
def test_real_misfires_are_flagged(text, language):
    assert unexpected_language(text, ALLOWED) == language


@pytest.mark.parametrize(
    "text",
    [
        "Okay.",
        "Thanks.",
        "Alfred, so",
        "Posaduczam intękną.",
        "What time is it?",
        "Tell me a joke.",
        "Hello Alfred",
        "Set a timer for five minutes.",
        "How's the weather outside in Kložnapoca?",
        "Da, dă mai multe detalii te rog.",
        "Caută cea mai vândută mașină din România în anul 2026.",
    ],
)
def test_english_and_romanian_pass(text):
    assert unexpected_language(text, ALLOWED) is None


@pytest.mark.parametrize(
    "text, language",
    [
        ("I think LeBron won a title with Cleveland too, are you sure?", "English"),
        ("Câte titluri a câștigat LeBron James și cu ce echipe?", "Romanian"),
        ("Okay.", None),
    ],
)
def test_spoken_language(text, language):
    assert spoken_language(text) == language
