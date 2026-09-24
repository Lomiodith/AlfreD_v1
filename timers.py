import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    id: int
    label: str
    due: float
    handle: threading.Timer = field(repr=False)

    def remaining_seconds(self) -> int:
        return max(0, round(self.due - time.time()))


class TimerService:
    """Countdown timers. Expiry is queued rather than spoken from the timer
    thread, so the announcement can't collide with speech already playing."""

    def __init__(self):
        self._timers: Dict[int, Timer] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self.announcements: "queue.Queue[str]" = queue.Queue()

    def start(self, seconds: int, label: str = "") -> Timer:
        with self._lock:
            timer_id = self._next_id
            self._next_id += 1
            handle = threading.Timer(seconds, self._fire, args=(timer_id,))
            handle.daemon = True
            timer = Timer(
                timer_id, label or f"{seconds} second", time.time() + seconds, handle
            )
            self._timers[timer_id] = timer
        handle.start()
        logger.info(f"⏲️ Timer {timer_id} '{timer.label}' set for {seconds}s")
        return timer

    def cancel(self, label_or_id: str) -> Optional[Timer]:
        with self._lock:
            timer = self._find(label_or_id)
            if timer:
                timer.handle.cancel()
                del self._timers[timer.id]
        return timer

    def active(self) -> List[Timer]:
        with self._lock:
            return sorted(self._timers.values(), key=lambda t: t.due)

    def cancel_all(self):
        with self._lock:
            for timer in self._timers.values():
                timer.handle.cancel()
            self._timers.clear()

    def pending_announcements(self) -> List[str]:
        messages = []
        while not self.announcements.empty():
            messages.append(self.announcements.get_nowait())
        return messages

    def _find(self, label_or_id: str) -> Optional[Timer]:
        key = str(label_or_id).strip().lower()
        for timer in self._timers.values():
            if key in (str(timer.id), timer.label.lower()):
                return timer
        return None

    def _fire(self, timer_id: int):
        with self._lock:
            timer = self._timers.pop(timer_id, None)
        if timer:
            logger.info(f"⏰ Timer {timer_id} '{timer.label}' finished")
            self.announcements.put(f"Your {timer.label} timer is done.")
