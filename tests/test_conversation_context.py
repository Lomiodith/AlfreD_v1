from unittest.mock import MagicMock, patch

import conversation_handler as ch
from memory_manager import FACT_EVENT, EpisodicMemory


def _memory(event_type, content="", user_input=""):
    return EpisodicMemory("id", 0, event_type, content, user_input, "", {})


def _handler(memories):
    manager = MagicMock()
    manager.search_memories.return_value = [(m, 0.8) for m in memories]
    with patch.object(ch, "MemoryManager", return_value=manager), patch.object(
        ch, "Embedder"
    ), patch.object(ch, "google_tools", return_value=[]), patch.object(
        ch, "github_tools", return_value=[]
    ):
        return ch.ConversationHandler(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )


def test_questions_already_in_this_conversation_are_not_recalled():
    handler = _handler(
        [
            _memory(
                "conversation_interaction", user_input="Who's the mayor of Bucharest?"
            ),
            _memory(
                "conversation_interaction", user_input="What's the weather in Cluj?"
            ),
        ]
    )
    handler.conversation_history.append(
        {"role": "user", "content": "Who's the mayor of Bucharest?"}
    )

    context = handler.get_relevant_context("Okay.")

    assert "mayor" not in context
    assert "weather in Cluj" in context


def test_past_answers_are_never_injected():
    answered = _memory("conversation_interaction", user_input="How many titles?")
    answered.assistant_response = "A wrong answer"
    handler = _handler([answered, _memory(FACT_EVENT, content="The user is 33")])

    context = handler.get_relevant_context("titles")

    assert "A wrong answer" not in context
    assert "The user is 33" in context and "How many titles?" in context


def test_key_press_cuts_the_answer_and_listens_again():
    from llm_service import AgentResult

    handler = _handler([])
    handler.voice_output = True
    tts = handler.tts_service
    tts.interrupted = False
    tts.reset.side_effect = lambda: setattr(tts, "interrupted", False)

    def run_agent(request, tools, execute, on_sentence, cancelled, **kwargs):
        on_sentence("Tomorrow you have one event.")
        tts.interrupted = True  # a key pressed during playback
        assert cancelled()
        on_sentence("It's an HR call.")  # generated after the key press: not spoken
        return AgentResult("Tomorrow you have one event. It's an HR call.")

    handler.llm_service.run_agent.side_effect = run_agent

    assert handler._dispatch_command("What's in my calendar tomorrow?")

    assert handler._awaiting_reply
    assert handler.conversation_history[-1] == {
        "role": "assistant",
        "content": "Tomorrow you have one event.",
    }
    tts.say.assert_called_once_with("Tomorrow you have one event.")
    handler.memory_manager.episodic_memory.assert_not_called()


def test_tool_calls_stay_in_the_history_with_trimmed_results():
    from llm_service import AgentResult

    handler = _handler([])
    handler.voice_output = False
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "type": "function", "function": {}}],
    }
    result = {"role": "tool", "tool_call_id": "1", "content": "x" * 5000}
    handler.llm_service.run_agent.return_value = AgentResult(
        "Opened it.", trail=[call, result]
    )

    handler._dispatch_command("Open the budget.")

    history = handler.conversation_history
    assert [m["role"] for m in history[-4:]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history[-3] is call
    assert len(history[-2]["content"]) == ch.HISTORY_TOOL_RESULT_CHARS + 1


def test_summarising_never_separates_a_tool_result_from_its_call():
    handler = _handler([])
    turn = [
        {"role": "user", "content": "Open it."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "opened"},
        {"role": "assistant", "content": "Opened." + " x" * 2000},
    ]
    handler.conversation_history += turn * 4
    handler.llm_service.get_completion.return_value = "summary"

    assert handler.auto_summarize_context()

    kept = [m for m in handler.conversation_history if m["role"] != "system"]
    assert kept[0]["role"] == "user"
