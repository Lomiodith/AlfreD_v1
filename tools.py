import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import file_opener
from memory_manager import FACT_EVENT

logger = logging.getLogger(__name__)

# Tool output goes back into the prompt; cap it so one big result can't blow
# the context window or the Groq token-per-minute limit.
MAX_RESULT_CHARS = 4000


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def params(required=(), **properties) -> Dict[str, Any]:
    """Build a JSON-schema object from keyword -> property-schema pairs."""
    return {"type": "object", "properties": properties, "required": list(required)}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, *tools: Tool):
        for tool in tools:
            self._tools[tool.name] = tool

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: str) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            kwargs = json.loads(arguments) if arguments else {}
            logger.info(f"🔧 {name}({kwargs})")
            result = tool.handler(**kwargs)
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            result = {"error": f"{type(e).__name__}: {e}"}
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return text[:MAX_RESULT_CHARS]


def core_tools(tool_manager, search_service, timer_service) -> List[Tool]:
    def web_search(query: str):
        items = search_service.search_ranked(query, num_results=5)
        return search_service.format_results(items, query) if items else "No results."

    def run_command(command: str):
        return tool_manager.execute_system_command(command)

    def read_file(path: str):
        return tool_manager.file_operations("read", path)

    def write_file(path: str, content: str):
        return tool_manager.file_operations("write", path, content)

    def open_file(name: str):
        return file_opener.open_file(name)

    def scrape_page(url: str, extract: str = "text"):
        return tool_manager.web_scraping(url, extract)

    def set_timer(seconds: int, label: str = ""):
        timer = timer_service.start(int(seconds), label)
        return {"id": timer.id, "label": timer.label, "seconds": int(seconds)}

    def list_timers():
        return [
            {"id": t.id, "label": t.label, "remaining_seconds": t.remaining_seconds()}
            for t in timer_service.active()
        ] or "No active timers."

    def cancel_timer(timer: str):
        cancelled = timer_service.cancel(timer)
        return (
            {"cancelled": cancelled.label} if cancelled else {"error": "No such timer"}
        )

    string = {"type": "string"}
    return [
        Tool(
            "web_search",
            "Search the web (DuckDuckGo). Use for current events, facts you're unsure of, "
            "or anything the user asks you to look up.",
            params(["query"], query=string),
            web_search,
        ),
        Tool(
            "run_command",
            "Run a bash command on the user's computer (Git Bash on Windows) and return its output. "
            "Destructive commands are blocked.",
            params(["command"], command=string),
            run_command,
        ),
        Tool(
            "read_file",
            "Read a text file's contents into this conversation, so you can summarise "
            "or answer questions about it. Only when the user wants you to read it; to "
            "open or show a file, use open_file.",
            params(["path"], path=string),
            read_file,
        ),
        Tool(
            "write_file",
            "Write text to a file on the user's computer, replacing its contents.",
            params(["path", "content"], path=string, content=string),
            write_file,
        ),
        Tool(
            "open_file",
            "Open a file or folder on the user's screen with its default program, as "
            "double-clicking it would. Use whenever the user says 'open', 'show' or "
            "'pull up' a file; don't ask whether to read it instead. `name` is a full path, or a file name as the "
            "user said it (e.g. 'config.py', 'budget', 'alfred tray'), which is searched "
            "for in their usual folders. If several match, ask the user which one.",
            params(["name"], name=string),
            open_file,
        ),
        Tool(
            "scrape_page",
            "Fetch a web page and extract its content.",
            params(
                ["url"],
                url=string,
                extract={
                    "type": "string",
                    "enum": ["text", "links", "images", "metadata", "all"],
                },
            ),
            scrape_page,
        ),
        Tool(
            "set_timer",
            "Start a countdown timer. Convert the requested duration to seconds.",
            params(
                ["seconds"],
                seconds={"type": "integer"},
                label={"type": "string", "description": "Short name, e.g. 'pasta'"},
            ),
            set_timer,
        ),
        Tool(
            "list_timers",
            "List running timers and time remaining.",
            params(),
            list_timers,
        ),
        Tool(
            "cancel_timer",
            "Cancel a running timer by its label or id.",
            params(["timer"], timer=string),
            cancel_timer,
        ),
    ]


def memory_tools(memory_manager) -> List[Tool]:
    def remember(fact: str):
        return {"saved": fact, "id": memory_manager.remember(fact)}

    def recall(query: str = ""):
        if not query.strip():
            return [
                {"id": m.id, "fact": m.content} for m in memory_manager.facts()
            ] or ("Nothing has been saved yet.")
        # Past questions only, never Alfred's past answers: a wrong answer
        # recalled here was repeated as fact instead of being searched again.
        return [
            (
                {"id": m.id, "fact": m.content}
                if m.event_type == FACT_EVENT
                else {"past question": m.user_input}
            )
            for m, _ in memory_manager.search_memories(
                query, limit=5, min_similarity=0.25
            )
        ] or f"Nothing relevant to '{query}' in memory."

    def forget(memory_id: str):
        return (
            {"forgotten": memory_id}
            if memory_manager.forget(memory_id)
            else ({"error": "No saved fact with that id; use recall to find it."})
        )

    return [
        Tool(
            "remember",
            "Save a fact for the long term. Only when the user explicitly asks you to "
            "remember something. Phrase it as a standalone statement, e.g. "
            "'The user's car is a 2019 Skoda Octavia.'",
            params(["fact"], fact={"type": "string"}),
            remember,
        ),
        Tool(
            "recall",
            "Look up long-term memory. With a query, search saved facts and the "
            "user's past questions by meaning (past answers are not kept: for facts "
            "about the world, use web_search). With no query, list every saved fact; "
            "use that for questions like 'what do you know about me?'.",
            params(query={"type": "string"}),
            recall,
        ),
        Tool(
            "forget",
            "Delete one saved fact by its id (get ids from recall).",
            params(["memory_id"], memory_id={"type": "string"}),
            forget,
        ),
    ]
