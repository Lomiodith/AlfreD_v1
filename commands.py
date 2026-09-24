"""Command reference shown by the "show commands" intent.

Keep in sync with INTENT_TRIGGERS: list only commands that are actually reachable.
"""

COMMAND_CATEGORIES = [
    (
        "🎯 Basic Commands",
        "Essential commands for controlling Alfred",
        [
            (
                "Alfred",
                "Wake word to activate Alfred",
                "Alfred (then wait for 'Wake word detected!')",
            ),
            ("close script", "Shut down Alfred completely", "Alfred, close script"),
            (
                "clear context",
                "Clear conversation history and start fresh",
                "Alfred, clear context",
            ),
            (
                "show commands",
                "Display this command reference",
                "Alfred, show commands",
            ),
            (
                "performance stats",
                "Show timing statistics for recent operations",
                "Alfred, performance stats",
            ),
        ],
    ),
    (
        "🔍 Search Commands",
        "Search the web and access information",
        [
            (
                "search for [query]",
                "Search the web via DuckDuckGo",
                "Alfred, search for python machine learning tutorials",
            ),
        ],
    ),
    (
        "💻 System Commands",
        "Run bash commands (Git Bash on Windows) with safety checks",
        [
            (
                "run command [command]",
                "Run a bash command and hear the result",
                "Alfred, run command git status",
            ),
        ],
    ),
    (
        "📁 File Operations",
        "Read and write files",
        [
            (
                "read file [path]",
                "Read the contents of a file",
                "Alfred, read file config.py",
            ),
            (
                "write file [path] with content [content]",
                "Write content to a file",
                "Alfred, write file notes.txt with content My important notes",
            ),
        ],
    ),
    (
        "🌐 Web Scraping",
        "Extract content from websites",
        [
            (
                "scrape [url]",
                "Extract text content from a webpage",
                "Alfred, scrape https://news.bbc.com",
            ),
            (
                "scrape [url] for [type]",
                "Extract text, links, images, metadata, or all",
                "Alfred, scrape https://example.com for links",
            ),
        ],
    ),
    (
        "💬 Conversation Features",
        "Natural conversation capabilities",
        [
            (
                "[any question or statement]",
                "Ask anything for natural conversation",
                "Alfred, what's the capital of France?",
            ),
            (
                "[follow-up question]",
                "Ask follow-up questions - Alfred remembers context",
                "Alfred, tell me more about that",
            ),
        ],
    ),
]

SAFETY_NOTES = [
    "🛡️ System commands are automatically checked for safety",
    "📁 File operations are restricted to safe directories in safe mode",
    "🌐 Web scraping only works with publicly accessible URLs",
    "⏱️ All operations have timeout protection (30 seconds for commands)",
    "💾 All interactions are automatically saved to memory for context",
]

TIPS_AND_TRICKS = [
    "💡 Alfred learns your preferences over time",
    "🔄 Context is automatically managed - old conversations are summarized when needed",
    "🎯 Be specific with file paths and URLs for better results",
    "💬 Alfred remembers previous conversations - reference them naturally",
    "🔍 Search results are cached for 1 hour, so repeat searches are instant",
    "⚡ Use simple, clear voice commands for best recognition",
]

RULE = "=" * 60


def _section(title, lines):
    return [title, "-" * len(title), *(f"  {line}" for line in lines), ""]


def get_formatted_commands():
    output = [RULE, "🤖 ALFRED COMMAND REFERENCE", RULE, ""]

    for title, description, commands in COMMAND_CATEGORIES:
        output += [title, "-" * len(title), description, ""]
        for command, what, example in commands:
            output += [
                f"  Command: {command}",
                f"  Description: {what}",
                f"  Example: {example}",
                "",
            ]

    output += _section("🛡️ SAFETY & LIMITATIONS", SAFETY_NOTES)
    output += _section("💡 TIPS & TRICKS", TIPS_AND_TRICKS)
    output += [RULE, "Say 'Alfred, [command]' to use any command above!", RULE]

    return "\n".join(output)
