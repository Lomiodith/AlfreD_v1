"""Command reference printed for "show commands".

The instant commands are generated from INTENT_TRIGGERS, so they can't drift
from what the detector actually accepts.
"""

from intent_detector import INTENT_TRIGGERS

INSTANT_COMMAND_DESCRIPTIONS = {
    "terminate": "Shut Alfred down",
    "clear_context": "Forget the current conversation (long-term memory stays)",
    "voice_off": "Text mode: full answers on screen, nothing spoken",
    "voice_on": "Voice mode: short spoken answers",
    "show_commands": "Show this reference",
    "performance": "Show timing statistics",
    "decline": "No: end the conversation (or, after 'Did you mean…?', say it again)",
    "stop": "Stop / that's all: end the conversation, whatever Alfred asked",
}

EXAMPLE_REQUESTS = [
    ("Web search", "What's the weather in Cluj this weekend?"),
    ("Gmail", "Do I have any unread emails from GitHub?"),
    ("Calendar", "What's on my calendar tomorrow? / Add dentist on Friday at 3"),
    ("GitHub", "Any GitHub notifications?"),
    ("Timers", "Set a timer for ten minutes for the pasta / Cancel the pasta timer"),
    ("Memory", "Remember that I'm 33 / What do you know about me? / Forget that"),
    ("Commands & files", "Run git status in bash / Read config.py"),
    ("Web pages", "Get the links from example.com"),
    ("Anything else", "Explain how a jet engine works"),
]

TIPS = [
    "Say 'Alfred', wait for 'Wake word detected', then just talk; it notices when you've finished",
    "After an answer, just keep talking without saying 'Alfred'; say 'no' or 'that's all' (or stay quiet) to end the conversation",
    "English and Romanian both work; Alfred replies in the language you used",
    "Press any key to interrupt Alfred while it's speaking",
    "Instant commands only trigger when they are the whole sentence",
]

RULE = "=" * 60


def _section(title, lines):
    return [title, "-" * len(title), *(f"  {line}" for line in lines), ""]


def get_formatted_commands():
    instant = [
        f"{' / '.join(INTENT_TRIGGERS[intent])}: {description}"
        for intent, description in INSTANT_COMMAND_DESCRIPTIONS.items()
    ]
    examples = [f'{area}: "{example}"' for area, example in EXAMPLE_REQUESTS]

    return "\n".join(
        [RULE, "🤖 ALFRED COMMAND REFERENCE", RULE, ""]
        + _section("💬 JUST ASK (the model picks the right tool)", examples)
        + _section("⚡ INSTANT COMMANDS (no model call)", instant)
        + _section("💡 TIPS", TIPS)
        + [RULE]
    )
