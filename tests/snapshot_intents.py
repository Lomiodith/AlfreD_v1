"""Characterisation snapshot for IntentDetector.

Pins how 49 sample utterances route today, so a refactor can prove it changed
nothing. This catches regressions; it does NOT tell you the pinned behaviour is
correct.

    python tests/snapshot_intents.py            compare against the baseline
    python tests/snapshot_intents.py --update   re-record it (only when the
                                                change is intended)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intent_detector import IntentDetector  # noqa: E402

BASELINE = Path(__file__).parent / "intent_baseline.json"

CASES = [
    "search for python tutorials",
    "look up the weather",
    "google cats",
    "find me a hotel",
    "what is quantum physics",
    "who is ada lovelace",
    "tell me about rome",
    "find out the time",
    "run command ls -la",
    "execute pwd",
    "terminal uname",
    "shell command dir",
    "read file config.py",
    "open file notes.txt",
    "show file a.md",
    "what's in file b.txt",
    "write file x.txt with content hello",
    "save file y",
    "create file z",
    "write to file q",
    "scrape https://a.com",
    "web scrape https://b.com for links",
    "get the page https://c.com",
    "fetch the page https://d.com",
    "grab the page https://e.com",
    "show commands",
    "help",
    "commands",
    "what can you do",
    "show me commands",
    "performance stats",
    "show stats",
    "performance",
    "how are you performing",
    "clear context",
    "reset context",
    "start fresh",
    "forget everything",
    "new conversation",
    "close script",
    "shut down",
    "goodbye",
    "turn off",
    "exit",
    "quit",
    "hello there how are you",
    "tell me a joke",
    "serch for python",
    "wat is the time",
]


def current():
    detector = IntentDetector()
    return {case: list(detector.detect(case)) for case in CASES}


def main():
    now = current()

    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps(now, indent=1), encoding="utf-8")
        print(f"baseline re-recorded: {len(now)} cases")
        return 0

    saved = json.loads(BASELINE.read_text(encoding="utf-8"))
    drift = {k: (saved[k], now[k]) for k in saved if saved[k] != now[k]}

    print(f"{len(saved) - len(drift)}/{len(saved)} cases match the baseline")
    for case, (before, after) in drift.items():
        print(f"  {case!r}\n     baseline: {before}\n     now:      {after}")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
