from difflib import SequenceMatcher

INTENT_TRIGGERS = {
    "search": (
        True,
        [
            "search for", "look up", "find me", "google",
            "search", "look for", "what is", "who is",
            "tell me about", "find out",
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
        ["show commands", "help", "commands", "what can you do", "show me commands"],
    ),
    "performance": (
        False,
        ["performance stats", "show stats", "how are you performing", "performance"],
    ),
    "clear_context": (
        False,
        [
            "clear context", "reset context", "start fresh",
            "forget everything", "new conversation",
        ],
    ),
    "terminate": (
        False,
        ["close script", "shut down", "goodbye", "turn off", "exit", "quit"],
    ),
}

FUZZY_THRESHOLD = 0.7


class IntentDetector:
    def __init__(self):
        # Flattened once at startup; longest triggers first so the most
        # specific phrase wins ("search for" before "search").
        self.triggers = sorted(
            (
                (trigger, intent, extract_query)
                for intent, (extract_query, triggers) in INTENT_TRIGGERS.items()
                for trigger in triggers
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )

    def detect(self, text):
        """Returns (intent_name, query) or ("general", original_text)."""
        normalized = text.lower().strip()

        # First pass: exact prefix match (fast, most reliable)
        for trigger, intent, extract_query in self.triggers:
            if normalized.startswith(trigger):
                query = text[len(trigger):].strip() if extract_query else text
                return intent, query

        # Second pass: trigger appears anywhere in the text
        for trigger, intent, _ in self.triggers:
            if trigger in normalized:
                return intent, text

        # Third pass: fuzzy similarity for close matches (speech-to-text errors)
        best_intent = None
        best_score = 0.0
        for trigger, intent, _ in self.triggers:
            input_start = " ".join(normalized.split()[: len(trigger.split())])
            score = SequenceMatcher(None, input_start, trigger).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score > FUZZY_THRESHOLD and best_intent:
            return best_intent, text

        return "general", text
