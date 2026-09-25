import re
from difflib import SequenceMatcher

# Instant control commands handled without the LLM. Everything else, including
# tool use (search, files, timers, email...), is the model's call.
INTENT_TRIGGERS = {
    "show_commands": [
        "show commands",
        "commands",
        "what can you do",
        "show me commands",
    ],
    "performance": ["performance stats", "show stats", "performance"],
    "clear_context": [
        "clear context",
        "reset context",
        "start fresh",
        "forget everything",
        "new conversation",
    ],
    # Both end the conversation (Alfred stops listening without the wake word)
    # with no LLM round trip. They differ only while Alfred's language question
    # is pending: "no" answers it (say it again), "stop" still ends everything.
    "decline": ["no", "nope", "no thanks", "nu"],
    "stop": [
        "stop",
        "that's all",
        "that's it",
        "i'm good",
        "nothing else",
        "enough",
        "gata",
    ],
    # Only meaningful while Alfred awaits a yes/no of its own (the language
    # check); otherwise "yes" goes to the LLM, e.g. as the answer to its follow-up question.
    "affirm": [
        "yes",
        "yeah",
        "yep",
        "sure",
        "correct",
        "that's right",
        "yes please",
        "da",
    ],
    "voice_on": ["voice mode", "voice on", "speak to me"],
    "voice_off": ["text mode", "voice off"],
    "terminate": ["close script", "shut down", "goodbye", "turn off", "exit", "quit"],
}

FUZZY_THRESHOLD = 0.85

# A control command must be the whole utterance, give or take these, so that
# "how do I exit vim" or "quit smoking tips" reach the LLM instead.
FILLER_WORDS = {
    "please",
    "now",
    "okay",
    "ok",
    "alfred",
    "hey",
    "thanks",
    "thank",
    "you",
}
NON_WORD = re.compile(r"[^\w\s']")

# A reply made only of these, with at least one of NO_WORDS or STOP_WORDS, is a
# decline or stop too: the whole-utterance rule sent "No, I said okay, stop." to
# the LLM, which went on. "Don't stop" or "stop the timer" still reach the LLM.
NO_WORDS = {"no", "nope", "nu"}
STOP_WORDS = {"stop", "enough", "gata", "destul"}
# "ai", "ay", "sed": "I said" as Parakeet spells it in Cyrillic ("ай сэд").
STOP_GLUE = {"i", "said", "just", "it", "that's", "all", "alright", "right"}
STOP_GLUE |= {"ai", "ay", "sed"}

# Parakeet guesses the language per utterance and writes a lone "stop" as
# "Стоп." Only English and Romanian are expected, so Cyrillic is misheard
# English: read it as Latin letters for command matching (the language guard
# still sees the original).
CYRILLIC = str.maketrans(
    {
        **dict(zip("абвгдезийклмнопрстуфхыэ", "abvgdeziiklmnoprstufhye")),
        **{"ё": "e", "ж": "zh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht"},
        **{"ю": "yu", "я": "ya", "ъ": "", "ь": ""},
    }
)


def _core(text: str) -> str:
    words = NON_WORD.sub(" ", text.lower().translate(CYRILLIC)).split()
    return " ".join(w for w in words if w not in FILLER_WORDS)


class IntentDetector:
    def __init__(self):
        # Triggers get the same filler-stripping as the input, or "what can you
        # do" could never match ("you" is filler).
        self.triggers = [
            (_core(trigger), intent)
            for intent, triggers in INTENT_TRIGGERS.items()
            for trigger in triggers
        ]
        self.exact = dict(self.triggers)

    def detect(self, text: str) -> str:
        """Return the control intent for `text`, or "general"."""
        core = _core(text)
        if core in self.exact:
            return self.exact[core]
        words = set(core.split())
        if (
            words & (NO_WORDS | STOP_WORDS)
            and words <= NO_WORDS | STOP_WORDS | STOP_GLUE
        ):
            return "stop" if words & STOP_WORDS else "decline"

        # Absorbs speech-to-text errors ("shut dawn").
        best_intent, best_score = "general", 0.0
        for trigger, intent in self.triggers:
            score = SequenceMatcher(None, core, trigger).ratio()
            if score > best_score:
                best_intent, best_score = intent, score

        return best_intent if best_score > FUZZY_THRESHOLD else "general"
