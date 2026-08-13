"""
Alfred Command Reference
========================

This file contains all available voice commands for Alfred.
Say "show commands" to display this help information.
"""

COMMAND_CATEGORIES = {
    "basic_commands": {
        "title": "🎯 Basic Commands",
        "description": "Essential commands for controlling Alfred",
        "commands": [
            {
                "command": "Alfred",
                "description": "Wake word to activate Alfred",
                "example": "Alfred (then wait for 'Wake word detected!')",
                "category": "wake"
            },
            {
                "command": "close script",
                "description": "Shut down Alfred completely",
                "example": "Alfred, close script",
                "category": "control"
            },
            {
                "command": "clear context",
                "description": "Clear conversation history and start fresh",
                "example": "Alfred, clear context",
                "category": "control"
            },
            {
                "command": "show commands",
                "description": "Display this command reference",
                "example": "Alfred, show commands",
                "category": "help"
            }
        ]
    },

    "search_commands": {
        "title": "🔍 Search Commands",
        "description": "Search the web and access information",
        "commands": [
            {
                "command": "search for [query]",
                "description": "Search the web using multiple sources with intelligent ranking",
                "example": "Alfred, search for python machine learning tutorials",
                "category": "search"
            },
            {
                "command": "search for [query] show thinking",
                "description": "Search and show the AI's reasoning process (DeepSeek R1)",
                "example": "Alfred, search for climate change show thinking",
                "category": "search"
            }
        ]
    },

    "system_commands": {
        "title": "💻 System Commands",
        "description": "Execute terminal/system commands safely",
        "commands": [
            {
                "command": "run command [command]",
                "description": "Execute a system command with safety checks",
                "example": "Alfred, run command ls -la",
                "category": "system"
            },
            {
                "command": "run command [command]",
                "description": "Check current directory contents",
                "example": "Alfred, run command pwd",
                "category": "system"
            },
            {
                "command": "run command [command]",
                "description": "Check system information",
                "example": "Alfred, run command uname -a",
                "category": "system"
            }
        ]
    },

    "file_commands": {
        "title": "📁 File Operations",
        "description": "Read, write, and manage files and directories",
        "commands": [
            {
                "command": "read file [path]",
                "description": "Read the contents of a file",
                "example": "Alfred, read file config.py",
                "category": "file"
            },
            {
                "command": "write file [path] with content [content]",
                "description": "Write content to a file",
                "example": "Alfred, write file notes.txt with content My important notes",
                "category": "file"
            },
            {
                "command": "file list [directory]",
                "description": "List contents of a directory",
                "example": "Alfred, file list /home/user/documents",
                "category": "file"
            },
            {
                "command": "file exists [path]",
                "description": "Check if a file or directory exists",
                "example": "Alfred, file exists config.json",
                "category": "file"
            },
            {
                "command": "file create [path]",
                "description": "Create a new directory",
                "example": "Alfred, file create new_project_folder",
                "category": "file"
            },
            {
                "command": "file delete [path]",
                "description": "Delete a file or empty directory",
                "example": "Alfred, file delete old_file.txt",
                "category": "file"
            }
        ]
    },

    "web_commands": {
        "title": "🌐 Web Scraping",
        "description": "Extract content from websites",
        "commands": [
            {
                "command": "scrape [url]",
                "description": "Extract text content from a webpage",
                "example": "Alfred, scrape https://news.bbc.com",
                "category": "web"
            },
            {
                "command": "scrape [url] for [type]",
                "description": "Extract specific content type (text, links, images, metadata, all)",
                "example": "Alfred, scrape https://example.com for links",
                "category": "web"
            },
            {
                "command": "web scrape [url] for [type]",
                "description": "Alternative syntax for web scraping",
                "example": "Alfred, web scrape https://github.com for metadata",
                "category": "web"
            }
        ]
    },

    "advanced_commands": {
        "title": "🧠 Advanced Features",
        "description": "Advanced AI capabilities and memory features",
        "commands": [
            {
                "command": "[any question] show thinking",
                "description": "Show the AI's reasoning process for any response",
                "example": "Alfred, explain quantum physics show thinking",
                "category": "advanced"
            },
            {
                "command": "[any question] show thoughts",
                "description": "Alternative to 'show thinking'",
                "example": "Alfred, what's the weather like show thoughts",
                "category": "advanced"
            },
            {
                "command": "enable tools",
                "description": "Enable GPT-OSS-120B internal tool calls (search, etc.)",
                "example": "Alfred, enable tools",
                "category": "advanced"
            },
            {
                "command": "disable tools",
                "description": "Disable GPT-OSS tool calls, use only AlfreD's search",
                "example": "Alfred, disable tools",
                "category": "advanced"
            },
            {
                "command": "toggle tools",
                "description": "Switch between GPT-OSS tools and AlfreD search",
                "example": "Alfred, toggle tools",
                "category": "advanced"
            }
        ]
    },

    "conversation_commands": {
        "title": "💬 Conversation Features",
        "description": "Natural conversation capabilities",
        "commands": [
            {
                "command": "[any question or statement]",
                "description": "Ask any question or make any statement for natural conversation",
                "example": "Alfred, what's the capital of France?",
                "category": "conversation"
            },
            {
                "command": "[follow-up question]",
                "description": "Ask follow-up questions - Alfred remembers context",
                "example": "Alfred, tell me more about that",
                "category": "conversation"
            },
            {
                "command": "[task request]",
                "description": "Request help with tasks or explanations",
                "example": "Alfred, help me write a Python function",
                "category": "conversation"
            }
        ]
    }
}

SAFETY_NOTES = [
    "🛡️ System commands are automatically checked for safety",
    "📁 File operations are restricted to safe directories in safe mode",
    "🌐 Web scraping only works with publicly accessible URLs",
    "⏱️ All operations have timeout protection (30 seconds for commands)",
    "💾 All interactions are automatically saved to memory for context"
]

TIPS_AND_TRICKS = [
    "💡 Alfred learns your preferences over time",
    "🔄 Context is automatically managed - old conversations are summarized when needed",
    "🎯 Be specific with file paths and URLs for better results",
    "📝 Use 'show thinking' to understand how DeepSeek R1 processes your requests",
    "💬 Alfred remembers previous conversations - reference them naturally",
    "🔍 Search results are cached for 1 hour to save API calls",
    "⚡ Use simple, clear voice commands for best recognition"
]

def get_formatted_commands():
    """Return formatted command reference as a string."""
    output = []
    output.append("=" * 60)
    output.append("🤖 ALFRED COMMAND REFERENCE")
    output.append("=" * 60)
    output.append("")

    for category_key, category in COMMAND_CATEGORIES.items():
        output.append(f"{category['title']}")
        output.append("-" * len(category['title']))
        output.append(f"{category['description']}")
        output.append("")

        for cmd in category['commands']:
            output.append(f"  Command: {cmd['command']}")
            output.append(f"  Description: {cmd['description']}")
            output.append(f"  Example: {cmd['example']}")
            output.append("")

    output.append("🛡️ SAFETY & LIMITATIONS")
    output.append("-" * 25)
    for note in SAFETY_NOTES:
        output.append(f"  {note}")
    output.append("")

    output.append("💡 TIPS & TRICKS")
    output.append("-" * 15)
    for tip in TIPS_AND_TRICKS:
        output.append(f"  {tip}")
    output.append("")

    output.append("=" * 60)
    output.append("Say 'Alfred, [command]' to use any command above!")
    output.append("=" * 60)

    return "\n".join(output)
