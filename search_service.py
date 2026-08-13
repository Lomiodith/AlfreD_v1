import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

from ddgs import DDGS

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

    def search(self, query, num_results=5):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            return {"items": [
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]}
        except Exception as e:
            print(f"Error during search: {e}")
            return None

    def _get_cache_file_path(self, query: str, num_results: int) -> str:
        # Built-in hash() is salted per process, so it would never produce a
        # cache hit across runs; md5 keeps keys stable on disk.
        digest = hashlib.md5(query.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}_{num_results}.json")

    def cached_search(self, query: str, num_results: int = 5,
                      cache_duration: int = 3600) -> Optional[Dict]:
        cache_file = self._get_cache_file_path(query, num_results)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                if time.time() - cached_data.get('timestamp', 0) < cache_duration:
                    print(f"Using cached results for: {query}")
                    return cached_data.get('results')
            except Exception as e:
                print(f"Error reading cache: {e}")

        results = self.search(query, num_results)

        if results:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(
                        {
                            'timestamp': time.time(),
                            'query': query,
                            'results': results
                        },
                        f
                    )
            except Exception as e:
                print(f"Error writing to cache: {e}")

        return results

    def search_results_ranking(self, results: Dict, query: str) -> Dict:
        if not results or "items" not in results:
            return results

        query_terms = set(query.lower().split())

        def relevance_score(item):
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()

            title_matches = sum(1 for term in query_terms if term in title)
            snippet_matches = sum(1 for term in query_terms if term in snippet)

            score = title_matches * 3 + snippet_matches

            title_word_count = len(title.split())
            if title_word_count:
                score += (title_matches / title_word_count) * 2

            return score

        results["items"] = sorted(
            results.get("items", []), key=relevance_score, reverse=True
        )
        return results

    def multi_source_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        search_results = {
            'duckduckgo': None,
            'combined_results': [],
            'sources_used': []
        }

        try:
            ddg_results = self.cached_search(query, num_results)
            if ddg_results:
                search_results['duckduckgo'] = ddg_results
                search_results['sources_used'].append('duckduckgo')

                for item in ddg_results.get('items', []):
                    item['source'] = 'duckduckgo'
                    search_results['combined_results'].append(item)

        except Exception as e:
            print(f"Error in DuckDuckGo search: {e}")

        if search_results['combined_results']:
            ranked = self.search_results_ranking(
                {'items': search_results['combined_results']}, query
            )
            search_results['combined_results'] = ranked.get('items', [])

        return search_results

    def format_multi_source_results(
        self, search_results: Dict[str, Any], query: str
    ) -> str:
        combined_results = search_results.get('combined_results', [])

        if not combined_results:
            return f"No search results found for '{query}'"

        lines = [SEARCH_PREAMBLE.format(query=query)]
        for idx, item in enumerate(combined_results, 1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Snippet").replace("\n", " ")
            lines.append(f"[{idx}] {title}\n    URL: {link}\n    {snippet}")

        return "\n".join(lines)
