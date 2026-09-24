import hashlib

import numpy as np
import pytest

from memory_manager import FACT_EVENT, MemoryManager


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: shared words mean high similarity."""

    def embed(self, texts):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(FakeEmbedder(), memory_dir=str(tmp_path))


def test_remember_then_list_facts(memory):
    memory.remember("The user's car is a Skoda Octavia")
    memory.remember("The user lives in Cluj")

    assert [m.content for m in memory.facts()] == [
        "The user lives in Cluj",
        "The user's car is a Skoda Octavia",
    ]


def test_facts_and_conversations_are_both_searchable(memory):
    memory.remember("The user's car is a Skoda Octavia")
    memory.episodic_memory(
        "conversation_interaction",
        "User: best pasta recipe\nAssistant: carbonara",
        user_input="best pasta recipe",
        assistant_response="carbonara",
    )

    car = memory.search_memories("skoda octavia car", min_similarity=0.3)
    pasta = memory.search_memories("pasta recipe", min_similarity=0.3)

    assert car[0][0].event_type == FACT_EVENT
    assert pasta[0][0].user_input == "best pasta recipe"


def test_summaries_are_not_searchable(memory):
    memory.episodic_memory("context_summarization", "summary about skoda cars")

    assert memory.search_memories("skoda cars", min_similarity=0.0) == []


def test_forget_removes_fact_from_list_and_search(memory):
    fact_id = memory.remember("The user's car is a Skoda Octavia")

    assert memory.forget(fact_id)
    assert memory.facts() == []
    assert memory.search_memories("skoda octavia", min_similarity=0.0) == []


def test_forget_refuses_conversation_history(memory):
    conversation_id = memory.episodic_memory(
        "conversation_interaction", "User: hi\nAssistant: hello", user_input="hi"
    )

    assert not memory.forget(conversation_id)


def test_index_survives_restart(tmp_path):
    MemoryManager(FakeEmbedder(), memory_dir=str(tmp_path)).remember("likes jazz")

    reloaded = MemoryManager(FakeEmbedder(), memory_dir=str(tmp_path))

    assert reloaded.search_memories("likes jazz")[0][0].content == "likes jazz"


def test_facts_saved_in_the_same_millisecond_are_both_kept(memory, monkeypatch):
    import memory_manager

    monkeypatch.setattr(memory_manager.time, "time", lambda: 1_700_000_000.0)
    memory.remember("The user's car is a Skoda")
    memory.remember("The user lives in Cluj")

    assert len(memory.facts()) == 2


def test_recall_returns_past_questions_but_never_past_answers(memory):
    from tools import ToolRegistry, memory_tools

    memory.episodic_memory(
        "conversation_interaction",
        "User: how many titles did LeBron win\nAssistant: 4, never with Cleveland",
        user_input="how many titles did LeBron win",
        assistant_response="4, never with Cleveland",
    )
    memory.remember("The user likes LeBron")
    registry = ToolRegistry()
    registry.register(*memory_tools(memory))

    result = registry.call("recall", '{"query": "LeBron titles"}')

    assert "how many titles did LeBron win" in result
    assert "The user likes LeBron" in result
    assert "Cleveland" not in result
