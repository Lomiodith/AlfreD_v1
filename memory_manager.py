import logging
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from config import MEMORY_MIN_SIMILARITY

logger = logging.getLogger(__name__)

MEMORY_COLUMNS = (
    "id, timestamp, event_type, content, user_input, assistant_response, context"
)
FACT_EVENT = "user_fact"
# Summaries aren't searchable: they have no user input to show as context.
SEARCHABLE_EVENTS = ("conversation_interaction", FACT_EVENT)
_SEARCHABLE_SQL = f"event_type IN ({', '.join('?' * len(SEARCHABLE_EVENTS))})"

TECH_KEYWORDS = ("python", "code", "programming", "software", "api", "database")
SCIENCE_KEYWORDS = ("research", "study", "experiment", "theory", "analysis")
CREATIVE_KEYWORDS = ("write", "story", "creative", "art", "design", "music")
BRIEF_KEYWORDS = ("brief", "short", "concise")
DETAILED_KEYWORDS = ("detailed", "explain", "elaborate")
FORMAL_INDICATORS = ("please", "thank you", "could you", "would you")
CASUAL_INDICATORS = ("hey", "yo", "what's up", "cool", "awesome")


@dataclass
class EpisodicMemory:
    id: str
    timestamp: float
    event_type: str
    content: str
    user_input: str
    assistant_response: str
    context: Dict[str, Any]


@dataclass
class UserPreference:
    category: str
    preference_type: str
    value: Any
    confidence: float
    last_updated: float
    frequency: int


class MemoryManager:
    def __init__(self, embedder, memory_dir="memory_data"):
        self.embedder = embedder
        self.memory_dir = memory_dir
        self.db_path = os.path.join(memory_dir, "memory.db")
        self.preferences_file = os.path.join(memory_dir, "user_preferences.json")
        self.user_preferences = {}
        self._preferences_loaded = False
        self._index_ids: List[str] = []
        self._index_vectors = np.zeros((0, 0), dtype=np.float32)
        self._setup_memory_storage()
        self._backfill_embeddings()
        self._load_index()

    @contextmanager
    def _connect(self):
        """Yield a cursor, committing on success and always closing the connection."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    def _setup_memory_storage(self):
        os.makedirs(self.memory_dir, exist_ok=True)

        with self._connect() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    content TEXT,
                    user_input TEXT,
                    assistant_response TEXT,
                    context TEXT,
                    embedding BLOB
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp "
                "ON episodic_memories(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type "
                "ON episodic_memories(event_type)"
            )
            columns = {
                row[1] for row in cursor.execute("PRAGMA table_info(episodic_memories)")
            }
            if "embedding" not in columns:
                cursor.execute(
                    "ALTER TABLE episodic_memories ADD COLUMN embedding BLOB"
                )

    def _backfill_embeddings(self):
        with self._connect() as cursor:
            rows = cursor.execute(
                "SELECT id, content FROM episodic_memories "
                f"WHERE {_SEARCHABLE_SQL} AND embedding IS NULL",
                SEARCHABLE_EVENTS,
            ).fetchall()
            if not rows:
                return
            logger.info(f"🧬 Embedding {len(rows)} existing memories (one-time)...")
            vectors = self.embedder.embed([content for _, content in rows])
            cursor.executemany(
                "UPDATE episodic_memories SET embedding = ? WHERE id = ?",
                [
                    (vec.tobytes(), memory_id)
                    for (memory_id, _), vec in zip(rows, vectors)
                ],
            )

    def _load_index(self):
        with self._connect() as cursor:
            rows = cursor.execute(
                "SELECT id, embedding FROM episodic_memories "
                f"WHERE {_SEARCHABLE_SQL} AND embedding IS NOT NULL",
                SEARCHABLE_EVENTS,
            ).fetchall()
        self._index_ids = [memory_id for memory_id, _ in rows]
        self._index_vectors = np.array(
            [np.frombuffer(blob, dtype=np.float32) for _, blob in rows],
            dtype=np.float32,
        )

    def _load_user_preferences(self):
        if self._preferences_loaded:
            return

        self.user_preferences = {}
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, "r") as f:
                    prefs_data = json.load(f)
                self.user_preferences = {
                    key: UserPreference(**data) for key, data in prefs_data.items()
                }
            except Exception as e:
                logger.error(f"Error loading user preferences: {e}")

        self._preferences_loaded = True

    def _save_user_preferences(self):
        try:
            prefs_data = {
                key: asdict(pref) for key, pref in self.user_preferences.items()
            }
            with open(self.preferences_file, "w") as f:
                json.dump(prefs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    def episodic_memory(
        self,
        event_type: str,
        content: str,
        user_input: str = "",
        assistant_response: str = "",
        context: Dict[str, Any] = None,
    ) -> str:
        if event_type == "conversation_interaction" and len(content) > 1000:
            content = content[:1000] + "..."

        # The random suffix matters: the model often saves two facts in one
        # turn, within the same millisecond, and a clash silently drops one.
        memory_id = f"{event_type}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        vector = (
            self.embedder.embed([content])[0]
            if event_type in SEARCHABLE_EVENTS
            else None
        )

        try:
            with self._connect() as cursor:
                cursor.execute(
                    f"INSERT INTO episodic_memories ({MEMORY_COLUMNS}, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        time.time(),
                        event_type,
                        content,
                        user_input,
                        assistant_response,
                        json.dumps(context or {}),
                        vector.tobytes() if vector is not None else None,
                    ),
                )
            if vector is not None:
                self._index_ids.append(memory_id)
                self._index_vectors = (
                    np.vstack([self._index_vectors, vector])
                    if self._index_vectors.size
                    else vector[None, :]
                )
            return memory_id

        except Exception as e:
            logger.error(f"Error storing episodic memory: {e}")
            return ""

    def search_memories(
        self, query: str, limit: int = 2, min_similarity: float = MEMORY_MIN_SIMILARITY
    ) -> List[Tuple[EpisodicMemory, float]]:
        if not self._index_ids:
            return []

        try:
            scores = self._index_vectors @ self.embedder.embed([query])[0]
            best = [
                i for i in np.argsort(-scores)[:limit] if scores[i] >= min_similarity
            ]
            if not best:
                return []

            ids = [self._index_ids[i] for i in best]
            with self._connect() as cursor:
                rows = cursor.execute(
                    f"SELECT {MEMORY_COLUMNS} FROM episodic_memories "
                    f"WHERE id IN ({', '.join('?' * len(ids))})",
                    ids,
                ).fetchall()
            by_id = {row[0]: row for row in rows}

            return [
                (self._row_to_memory(by_id[self._index_ids[i]]), float(scores[i]))
                for i in best
                if self._index_ids[i] in by_id
            ]

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def remember(self, fact: str) -> str:
        return self.episodic_memory(FACT_EVENT, content=fact.strip())

    def facts(self, limit: int = 50) -> List[EpisodicMemory]:
        with self._connect() as cursor:
            rows = cursor.execute(
                f"SELECT {MEMORY_COLUMNS} FROM episodic_memories "
                "WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (FACT_EVENT, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def forget(self, memory_id: str) -> bool:
        """Delete one saved fact. Conversation history can't be deleted this way."""
        with self._connect() as cursor:
            cursor.execute(
                "DELETE FROM episodic_memories WHERE id = ? AND event_type = ?",
                (memory_id, FACT_EVENT),
            )
            deleted = cursor.rowcount > 0
        if deleted:
            self._load_index()
        return deleted

    @staticmethod
    def _row_to_memory(row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row[0],
            timestamp=row[1],
            event_type=row[2],
            content=row[3],
            user_input=row[4],
            assistant_response=row[5],
            context=json.loads(row[6]) if row[6] else {},
        )

    def user_preference_learning(self, interaction_data: Dict[str, Any]):
        try:
            self._load_user_preferences()

            self._analyze_response_preferences(interaction_data)
            self._analyze_topic_preferences(interaction_data)
            self._analyze_communication_style(interaction_data)
            self._analyze_search_patterns(interaction_data)

            self._save_user_preferences()

        except Exception as e:
            logger.error(f"Error in user preference learning: {e}")

    def _analyze_response_preferences(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get("user_input", "").lower()

        if any(word in user_input for word in BRIEF_KEYWORDS):
            self._update_preference("response_style", "length_preference", "brief", 0.7)
        elif any(word in user_input for word in DETAILED_KEYWORDS):
            self._update_preference(
                "response_style", "length_preference", "detailed", 0.7
            )

    def _analyze_topic_preferences(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get("user_input", "").lower()

        topics = (
            ("technology_interest", TECH_KEYWORDS),
            ("science_interest", SCIENCE_KEYWORDS),
            ("creative_interest", CREATIVE_KEYWORDS),
        )
        for preference_type, keywords in topics:
            if any(keyword in user_input for keyword in keywords):
                self._update_preference("topics", preference_type, True, 0.6)

    def _analyze_communication_style(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get("user_input", "").lower()

        if any(indicator in user_input for indicator in FORMAL_INDICATORS):
            self._update_preference("communication", "formality_level", "formal", 0.5)
        elif any(indicator in user_input for indicator in CASUAL_INDICATORS):
            self._update_preference("communication", "formality_level", "casual", 0.5)

    def _analyze_search_patterns(self, interaction_data: Dict[str, Any]):
        if interaction_data.get("search_performed"):
            search_query = interaction_data.get("search_query", "")
            if search_query:
                self._update_preference(
                    "search", "frequent_search_topics", search_query, 0.3
                )

    def _update_preference(
        self, category: str, preference_type: str, value: Any, confidence: float
    ):
        key = f"{category}_{preference_type}"
        current_time = time.time()

        pref = self.user_preferences.get(key)
        if pref:
            pref.value = value
            pref.confidence = min(1.0, pref.confidence + confidence * 0.1)
            pref.last_updated = current_time
            pref.frequency += 1
        else:
            self.user_preferences[key] = UserPreference(
                category=category,
                preference_type=preference_type,
                value=value,
                confidence=confidence,
                last_updated=current_time,
                frequency=1,
            )
