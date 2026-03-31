import time
import json
import os
from typing import Dict, List, Any, Optional
from ddgs import DDGS


class SearchService:
    def __init__(self, cache_dir="search_cache"):
        self.cache_dir = cache_dir
        self._setup_cache_directory()

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

    def format_search_results(self, results, query):
        if not results or "items" not in results:
            return f"No search results found for '{query}'"

        formatted_results = []
        for idx, item in enumerate(results.get("items", []), 1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Snippet").replace("\n", " ")
            formatted_results.append(
                f"{idx}. Title: {title}\n   Link: {link}\n   Snippet: {snippet}"
            )

        return f"Search results for '{query}':\n" + "\n".join(formatted_results)

    def _setup_cache_directory(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_key(self, query: str, num_results: int = 5) -> str:
        return f"{hash(query)}_{num_results}"

    def _get_cache_file_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def cached_search(self, query: str, num_results: int = 5, cache_duration: int = 3600) -> Optional[Dict]:
        cache_key = self._get_cache_key(query, num_results)
        cache_file = self._get_cache_file_path(cache_key)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < cache_duration:
                    print(f"Using cached results for: {query}")
                    return cached_data.get('results')
            except Exception as e:
                print(f"Error reading cache: {e}")

        results = self.search(query, num_results)

        if results:
            try:
                cache_data = {
                    'timestamp': time.time(),
                    'query': query,
                    'results': results
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                print(f"Error writing to cache: {e}")

        return results

    def search_results_ranking(self, results: Dict, query: str) -> Dict:
        if not results or "items" not in results:
            return results

        items = results.get("items", [])
        query_terms = set(query.lower().split())

        def calculate_relevance_score(item):
            score = 0
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()

            title_matches = sum(1 for term in query_terms if term in title)
            snippet_matches = sum(1 for term in query_terms if term in snippet)

            score += title_matches * 3
            score += snippet_matches * 1

            title_word_count = len(title.split())
            if title_word_count > 0:
                score += (title_matches / title_word_count) * 2

            return score

        sorted_items = sorted(items, key=calculate_relevance_score, reverse=True)
        results["items"] = sorted_items

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
            ranked_results = {'items': search_results['combined_results']}
            ranked_results = self.search_results_ranking(ranked_results, query)
            search_results['combined_results'] = ranked_results.get('items', [])

        return search_results

    def format_multi_source_results(self, search_results: Dict[str, Any], query: str) -> str:
        combined_results = search_results.get('combined_results', [])

        if not combined_results:
            return f"No search results found for '{query}'"

        formatted_results = []
        formatted_results.append(f"Use the following web search results to answer the user's question about '{query}'. Summarize the key information from the snippets into a helpful, conversational answer. Cite specific products, names, or recommendations where available.\n")

        for idx, item in enumerate(combined_results, 1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Snippet").replace("\n", " ")

            formatted_results.append(
                f"[{idx}] {title}\n    URL: {link}\n    {snippet}"
            )

        return "\n".join(formatted_results)
