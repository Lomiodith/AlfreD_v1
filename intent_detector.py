import re
from difflib import SequenceMatcher

INTENT_TRIGGERS = {
    "search": (
        True,
        [
            "search for",
            "look up",
            "find me",
            "google",
            "search",
            "look for",
            "what is",
            "who is",
            "tell me about",
            "find out",
        ],
    ),
    "system_command": (
        True,
        ["run command", "execute", "run the command", "terminal", "shell command"],
    ),
    "read_file": (
        True,
        ["read file", "open file", "show file", "read the file", "what's in file"],
    ),
    "write_file": (
        True,
        ["write file", "save file", "create file", "write to file"],
    ),
    "scrape": (
        True,
        ["scrape", "web scrape", "get the page", "fetch the page", "grab the page"],
    ),
    "show_commands": (
        False,
        ["show commands", "commands", "what can you do", "show me commands"],
    ),
    "performance": (
        False,
        ["performance stats", "show stats", "how are you performing", "performance"],
    ),
    "clear_context": (
        False,
        [
            "clear context",
            "reset context",
            "start fresh",
            "forget everything",
            "new conversation",
        ],
    ),
    "terminate": (
        False,
        ["close script", "shut down", "goodbye", "turn off", "exit", "quit"],
    ),
}

FUZZY_THRESHOLD = 0.9

# Whisper punctuates transcripts ("Read file notes.txt."); none of it belongs in
# the argument.
QUERY_EDGE_PUNCTUATION = " ,.;:!?"


class IntentDetector:
    def __init__(self):
        # Flattened once at startup; longest triggers first so the most
        # specific phrase wins ("search for" before "search").
        self.triggers = sorted(
            (
                (
                    trigger,
                    intent,
                    extract_query,
                    re.compile(r"\b" + re.escape(trigger) + r"\b"),
                )
                for intent, (extract_query, triggers) in INTENT_TRIGGERS.items()
                for trigger in triggers
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def detect(self, text):
        """Returns (intent_name, query) or ("general", original_text)."""
        normalized = text.lower().strip()

        for trigger, intent, extract_query, pattern in self.triggers:
            if pattern.match(normalized):
                if extract_query:
                    return intent, text[len(trigger) :].strip(QUERY_EDGE_PUNCTUATION)
                return intent, text

        for _, intent, _, pattern in self.triggers:
            if pattern.search(normalized):
                return intent, text

        # Absorbs speech-to-text errors in the leading words ("serch for").
        best_intent = None
        best_score = 0.0
        for trigger, intent, _, _ in self.triggers:
            input_start = " ".join(normalized.split()[: len(trigger.split())])
            score = SequenceMatcher(None, input_start, trigger).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score > FUZZY_THRESHOLD and best_intent:
            return best_intent, text

        return "general", text
