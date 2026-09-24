import json
from unittest.mock import MagicMock

import pytest

from timers import TimerService
from tools import MAX_RESULT_CHARS, Tool, ToolRegistry, core_tools, params


@pytest.fixture
def registry():
    registry = ToolRegistry()
    registry.register(*core_tools(MagicMock(), MagicMock(), TimerService()))
    return registry


def test_schemas_are_openai_function_format(registry):
    schemas = registry.schemas()

    assert all(s["type"] == "function" for s in schemas)
    names = {s["function"]["name"] for s in schemas}
    assert {"web_search", "run_command", "set_timer", "cancel_timer"} <= names


def test_call_runs_handler_with_parsed_arguments(registry):
    result = json.loads(
        registry.call("set_timer", '{"seconds": 300, "label": "pasta"}')
    )

    assert result == {"id": 1, "label": "pasta", "seconds": 300}


def test_unknown_tool_and_handler_errors_come_back_as_json(registry):
    assert "Unknown tool" in json.loads(registry.call("nope", "{}"))["error"]
    assert "error" in json.loads(registry.call("set_timer", '{"wrong": 1}'))
    assert "error" in json.loads(registry.call("set_timer", "not json"))


def test_long_results_are_truncated():
    registry = ToolRegistry()
    registry.register(Tool("big", "", params(), lambda: "x" * (MAX_RESULT_CHARS * 2)))

    assert len(registry.call("big", "")) == MAX_RESULT_CHARS
