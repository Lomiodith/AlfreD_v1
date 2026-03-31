from difflib import SequenceMatcher


class IntentDetector:
    def __init__(self):
        self.intents = {
            "search": {
                "triggers": [
                    "search for", "look up", "find me", "google",
                    "search", "look for", "what is", "who is",
                    "tell me about", "find out"
                ],
                "extract_query": True,
            },
            "system_command": {
                "triggers": [
                    "run command", "execute", "run the command",
                    "terminal", "shell command"
                ],
                "extract_query": True,
            },
            "read_file": {
                "triggers": [
                    "read file", "open file", "show file",
                    "read the file", "what's in file"
                ],
                "extract_query": True,
            },
            "write_file": {
                "triggers": ["write file", "save file", "create file", "write to file"],
                "extract_query": True,
            },
            "scrape": {
                "triggers": [
                    "scrape", "web scrape", "get the page",
                    "fetch the page", "grab the page"
                ],
                "extract_query": True,
            },
            "show_commands": {
                "triggers": [
                    "show commands", "help", "commands",
                    "what can you do", "show me commands"
                ],
                "extract_query": False,
            },
            "performance": {
                "triggers": [
                    "performance stats", "show stats",
                    "how are you performing", "performance"
                ],
                "extract_query": False,
            },
            "clear_context": {
                "triggers": [
                    "clear context", "reset context", "start fresh",
                    "forget everything", "new conversation"
                ],
                "extract_query": False,
            },
            "terminate": {
                "triggers": [
                    "close script", "shut down", "goodbye",
                    "turn off", "exit", "quit"
                ],
                "extract_query": False,
            },
        }

    def detect(self, text):
        """Returns (intent_name, query) or ("general", original_text)."""
        normalized = text.lower().strip()

        # First pass: exact prefix match (fast, most reliable)
        for intent_name, config in self.intents.items():
            for trigger in sorted(config["triggers"], key=len, reverse=True):
                if normalized.startswith(trigger):
                    query = text[len(trigger):].strip() if config["extract_query"] else text
                    return intent_name, query

        # Second pass: trigger appears anywhere in the text
        for intent_name, config in self.intents.items():
            for trigger in config["triggers"]:
                if trigger in normalized:
                    return intent_name, text

        # Third pass: fuzzy similarity for close matches (handles speech-to-text errors)
        best_match = None
        best_score = 0.0
        for intent_name, config in self.intents.items():
            for trigger in config["triggers"]:
                input_start = " ".join(normalized.split()[:len(trigger.split())])
                score = SequenceMatcher(None, input_start, trigger).ratio()
                if score > best_score:
                    best_score = score
                    best_match = (intent_name, config)

        if best_score > 0.7 and best_match:
            intent_name, config = best_match
            return intent_name, text

        return "general", text
