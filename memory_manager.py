import json
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

MEMORY_COLUMNS = (
    "id, timestamp, event_type, content, user_input, assistant_response, "
    "context, embedding"
)

TECH_KEYWORDS = ('python', 'code', 'programming', 'software', 'api', 'database')
SCIENCE_KEYWORDS = ('research', 'study', 'experiment', 'theory', 'analysis')
CREATIVE_KEYWORDS = ('write', 'story', 'creative', 'art', 'design', 'music')
BRIEF_KEYWORDS = ('brief', 'short', 'concise')
DETAILED_KEYWORDS = ('detailed', 'explain', 'elaborate')
FORMAL_INDICATORS = ('please', 'thank you', 'could you', 'would you')
CASUAL_INDICATORS = ('hey', 'yo', "what's up", 'cool', 'awesome')


@dataclass
class EpisodicMemory:
    id: str
    timestamp: float
    event_type: str
    content: str
    user_input: str
    assistant_response: str
    context: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class UserPreference:
    category: str
    preference_type: str
    value: Any
    confidence: float
    last_updated: float
    frequency: int


class MemoryManager:
    def __init__(self, memory_dir="memory_data"):
        self.memory_dir = memory_dir
        self.db_path = os.path.join(memory_dir, "memory.db")
        self.preferences_file = os.path.join(memory_dir, "user_preferences.json")
        self.user_preferences = {}
        self._preferences_loaded = False
        self._setup_memory_storage()

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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    content TEXT,
                    user_input TEXT,
                    assistant_response TEXT,
                    context TEXT,
                    embedding TEXT
                )
            ''')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_timestamp '
                'ON episodic_memories(timestamp)'
            )
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_event_type '
                'ON episodic_memories(event_type)'
            )
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

    def _load_user_preferences(self) -> Dict[str, UserPreference]:
        if self._preferences_loaded:
            return self.user_preferences

        self.user_preferences = {}
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    prefs_data = json.load(f)
                self.user_preferences = {
                    key: UserPreference(**data) for key, data in prefs_data.items()
                }
            except Exception as e:
                print(f"Error loading user preferences: {e}")

        self._preferences_loaded = True
        return self.user_preferences

    def _save_user_preferences(self):
        try:
            prefs_data = {
                key: asdict(pref) for key, pref in self.user_preferences.items()
            }
            with open(self.preferences_file, 'w') as f:
                json.dump(prefs_data, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")

    def episodic_memory(self, event_type: str, content: str, user_input: str = "",
                        assistant_response: str = "",
                        context: Dict[str, Any] = None) -> str:
        if event_type == "conversation_interaction" and len(content) > 1000:
            content = content[:1000] + "..."

        memory_id = f"{event_type}_{int(time.time() * 1000)}"

        try:
            with self._connect() as cursor:
                cursor.execute(
                    f'INSERT INTO episodic_memories ({MEMORY_COLUMNS}) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        memory_id,
                        time.time(),
                        event_type,
                        content,
                        user_input,
                        assistant_response,
                        json.dumps(context or {}),
                        None
                    )
                )
            return memory_id

        except Exception as e:
            print(f"Error storing episodic memory: {e}")
            return ""

    def recall_episodic_memories(self, event_type: str = None, limit: int = 10,
                                 days_back: int = 30) -> List[EpisodicMemory]:
        cutoff_time = time.time() - (days_back * 24 * 3600)

        try:
            with self._connect() as cursor:
                if event_type:
                    cursor.execute(
                        f'SELECT {MEMORY_COLUMNS} FROM episodic_memories '
                        'WHERE event_type = ? AND timestamp > ? '
                        'ORDER BY timestamp DESC LIMIT ?',
                        (event_type, cutoff_time, limit)
                    )
                else:
                    cursor.execute(
                        f'SELECT {MEMORY_COLUMNS} FROM episodic_memories '
                        'WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?',
                        (cutoff_time, limit)
                    )
                rows = cursor.fetchall()

            return [
                EpisodicMemory(
                    id=row[0],
                    timestamp=row[1],
                    event_type=row[2],
                    content=row[3],
                    user_input=row[4],
                    assistant_response=row[5],
                    context=json.loads(row[6]) if row[6] else {},
                    embedding=json.loads(row[7]) if row[7] else None
                )
                for row in rows
            ]

        except Exception as e:
            print(f"Error recalling episodic memories: {e}")
            return []

    def user_preference_learning(self, interaction_data: Dict[str, Any]):
        try:
            self._load_user_preferences()

            self._analyze_response_preferences(interaction_data)
            self._analyze_topic_preferences(interaction_data)
            self._analyze_communication_style(interaction_data)
            self._analyze_search_patterns(interaction_data)

            self._save_user_preferences()

        except Exception as e:
            print(f"Error in user preference learning: {e}")

    def _analyze_response_preferences(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get('user_input', '').lower()

        if 'show thinking' in user_input or 'show thoughts' in user_input:
            self._update_preference(
                'response_style', 'show_thinking_preference', True, 0.8
            )

        if any(word in user_input for word in BRIEF_KEYWORDS):
            self._update_preference(
                'response_style', 'length_preference', 'brief', 0.7
            )
        elif any(word in user_input for word in DETAILED_KEYWORDS):
            self._update_preference(
                'response_style', 'length_preference', 'detailed', 0.7
            )

    def _analyze_topic_preferences(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get('user_input', '').lower()

        topics = (
            ('technology_interest', TECH_KEYWORDS),
            ('science_interest', SCIENCE_KEYWORDS),
            ('creative_interest', CREATIVE_KEYWORDS),
        )
        for preference_type, keywords in topics:
            if any(keyword in user_input for keyword in keywords):
                self._update_preference('topics', preference_type, True, 0.6)

    def _analyze_communication_style(self, interaction_data: Dict[str, Any]):
        user_input = interaction_data.get('user_input', '').lower()

        if any(indicator in user_input for indicator in FORMAL_INDICATORS):
            self._update_preference(
                'communication', 'formality_level', 'formal', 0.5
            )
        elif any(indicator in user_input for indicator in CASUAL_INDICATORS):
            self._update_preference(
                'communication', 'formality_level', 'casual', 0.5
            )

    def _analyze_search_patterns(self, interaction_data: Dict[str, Any]):
        if interaction_data.get('search_performed'):
            search_query = interaction_data.get('search_query', '')
            if search_query:
                self._update_preference(
                    'search', 'frequent_search_topics', search_query, 0.3
                )

    def _update_preference(self, category: str, preference_type: str,
                           value: Any, confidence: float):
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
                frequency=1
            )

    def get_user_preferences(self, category: str = None) -> Dict[str, UserPreference]:
        self._load_user_preferences()

        if category:
            return {
                key: pref for key, pref in self.user_preferences.items()
                if pref.category == category
            }
        return self.user_preferences.copy()

    def get_preference_summary(self) -> Dict[str, Any]:
        summary = defaultdict(dict)

        for pref in self.user_preferences.values():
            if pref.confidence > 0.5:
                summary[pref.category][pref.preference_type] = {
                    'value': pref.value,
                    'confidence': pref.confidence,
                    'frequency': pref.frequency
                }

        return dict(summary)

    def cleanup_old_memories(self, days_to_keep: int = 90):
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        try:
            with self._connect() as cursor:
                cursor.execute(
                    'DELETE FROM episodic_memories WHERE timestamp < ?', (cutoff_time,)
                )
                deleted_count = cursor.rowcount

            print(
                f"Cleaned up {deleted_count} old memories "
                f"(older than {days_to_keep} days)"
            )

        except Exception as e:
            print(f"Error cleaning up old memories: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        try:
            with self._connect() as cursor:
                cursor.execute('SELECT COUNT(*) FROM episodic_memories')
                total_memories = cursor.fetchone()[0]

                cursor.execute(
                    'SELECT event_type, COUNT(*) FROM episodic_memories '
                    'GROUP BY event_type'
                )
                memory_types = dict(cursor.fetchall())

                cursor.execute(
                    'SELECT MIN(timestamp), MAX(timestamp) FROM episodic_memories'
                )
                oldest, newest = cursor.fetchone()

            return {
                'total_memories': total_memories,
                'memory_types': memory_types,
                'oldest_memory': datetime.fromtimestamp(oldest) if oldest else None,
                'newest_memory': datetime.fromtimestamp(newest) if newest else None,
                'total_preferences': len(self.user_preferences)
            }

        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
