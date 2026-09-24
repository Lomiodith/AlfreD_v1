import logging
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

from ddgs import DDGS

logger = logging.getLogger(__name__)

CACHE_SECONDS = 3600
# Results go into every later request of an agent turn, so keep them lean:
# the per-minute token limit is what trips first.
MAX_SNIPPET_CHARS = 300

SEARCH_PREAMBLE = (
    "Use the following web search results to answer the user's question about "
    "'{query}'. Summarize the key information from the snippets into a helpful, "
    "conversational answer. Cite specific products, names, or recommendations "
    "where available.\n"
)


class SearchService:
    def __init__(self, cache_dir="search_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def search_ranked(self, query: str, num_results: int = 5) -> List[Dict]:
        items = self._cached_search(query, num_results) or []
        return self._rank(items, query)

    def format_results(self, items: List[Dict], query: str) -> str:
        lines = [SEARCH_PREAMBLE.format(query=query)]
        for idx, item in enumerate(items, 1):
            snippet = item["snippet"].replace("\n", " ")[:MAX_SNIPPET_CHARS]
            lines.append(
                f"[{idx}] {item['title']}\n    URL: {item['link']}\n    {snippet}"
            )
        return "\n".join(lines)

    def _search(self, query: str, num_results: int) -> Optional[List[Dict]]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return None

    def _cache_path(self, query: str, num_results: int) -> str:
        # Built-in hash() is salted per process, so it would never produce a
        # cache hit across runs; md5 keeps keys stable on disk.
        digest = hashlib.md5(query.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}_{num_results}.json")

    def _cached_search(self, query: str, num_results: int) -> Optional[List[Dict]]:
        cache_file = self._cache_path(query, num_results)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    time.time() - cached.get("timestamp", 0) < CACHE_SECONDS
                    and "items" in cached
                ):
                    logger.info(f"Using cached results for: {query}")
                    return cached["items"]
            except Exception as e:
                logger.error(f"Error reading cache: {e}")

        items = self._search(query, num_results)

        if items:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {"timestamp": time.time(), "query": query, "items": items}, f
                    )
            except Exception as e:
                logger.error(f"Error writing to cache: {e}")

        return items

    @staticmethod
    def _rank(items: List[Dict], query: str) -> List[Dict]:
        query_terms = set(query.lower().split())

        def relevance_score(item):
            title = item["title"].lower()
            snippet = item["snippet"].lower()

            title_matches = sum(1 for term in query_terms if term in title)
            snippet_matches = sum(1 for term in query_terms if term in snippet)
            score = title_matches * 3 + snippet_matches

            title_word_count = len(title.split())
            if title_word_count:
                score += (title_matches / title_word_count) * 2

            return score

        return sorted(items, key=relevance_score, reverse=True)
