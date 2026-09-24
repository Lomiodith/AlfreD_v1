import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

COMMAND_TIMEOUT = 30
REQUEST_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

DANGEROUS_COMMANDS = [
    "rm",
    "rmdir",
    "del",
    "format",
    "fdisk",
    "mkfs",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "chmod 777",
    "chown root",
]
FORBIDDEN_PATHS = ["/etc", "/sys", "/proc", "/dev", "/boot", "/root"]
FORBIDDEN_DOMAINS = ["localhost", "127.0.0.1", "0.0.0.0"]


def _find_bash() -> Optional[str]:
    if sys.platform != "win32":
        return shutil.which("bash")
    # A PATH lookup on Windows finds WSL's System32/bash.exe first; Git Bash
    # ships next to git, so locate it from there.
    git = shutil.which("git")
    if git:
        for parent in Path(git).parents:
            bash = parent / "bin" / "bash.exe"
            if bash.exists():
                return str(bash)
    return None


BASH = _find_bash()


class ToolManager:
    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode

    def execute_system_command(self, command: str) -> Dict[str, Any]:
        if not BASH:
            return {
                "success": False,
                "output": "",
                "error": "bash not found (on Windows, install Git for Windows)",
                "command": command,
            }

        try:
            if self.safe_mode:
                blocked_reason = self._unsafe_command_reason(command)
                if blocked_reason:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Command blocked for safety: {blocked_reason}",
                        "command": command,
                    }

            result = subprocess.run(
                [BASH, "-c", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=COMMAND_TIMEOUT,
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {COMMAND_TIMEOUT} seconds",
                "command": command,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Error executing command: {e}",
                "command": command,
            }

    def file_operations(
        self, action: str, path: str, content: str = None
    ) -> Dict[str, Any]:
        handlers = {
            "read": lambda: self._read_file(path),
            "write": lambda: self._write_file(path, content),
            "create": lambda: self._create_file_or_directory(path, content),
            "delete": lambda: self._delete_file_or_directory(path),
            "list": lambda: self._list_directory(path),
            "exists": lambda: self._check_exists(path),
        }

        handler = handlers.get(action.lower())
        if not handler:
            return {
                "success": False,
                "error": f'Unknown action: {action}. Available: {", ".join(handlers)}',
            }

        try:
            if self.safe_mode and not self._is_path_safe(path):
                return {
                    "success": False,
                    "error": f"Path not allowed in safe mode: {path}",
                }

            return handler()

        except Exception as e:
            return {"success": False, "error": f"File operation error: {e}"}

    def web_scraping(self, url: str, extract_type: str = "text") -> Dict[str, Any]:
        extractors = {
            "text": self._extract_text_content,
            "links": self._extract_links,
            "images": self._extract_images,
            "metadata": self._extract_metadata,
            "all": self._extract_all_content,
        }

        extractor = extractors.get(extract_type.lower())
        if not extractor:
            return {
                "success": False,
                "error": (
                    f"Unknown extract_type: {extract_type}. "
                    f'Available: {", ".join(extractors)}'
                ),
            }

        try:
            if not self._is_url_safe(url):
                return {
                    "success": False,
                    "error": "URL not allowed or potentially unsafe",
                }

            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            # Parsing is the expensive part, so do it once and share the tree.
            soup = BeautifulSoup(response.content, "html.parser")
            return extractor(soup, url)

        except requests.RequestException as e:
            return {"success": False, "error": f"Network error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Web scraping error: {e}"}

    def _unsafe_command_reason(self, command: str) -> str:
        """Return why a command is unsafe, or an empty string if it looks fine."""
        command_lower = command.lower().strip()

        for dangerous_cmd in DANGEROUS_COMMANDS:
            if re.search(rf"\b{re.escape(dangerous_cmd)}\b", command_lower):
                return f"Contains dangerous command: {dangerous_cmd}"

        if ">" in command and "/dev/" in command:
            return "Potential device file manipulation"

        if len(command) > 50 and any(c in command for c in ("|", ";", "&")):
            return "Complex command chaining detected"

        return ""

    def _is_path_safe(self, path: str) -> bool:
        path_lower = path.lower()
        if any(forbidden in path_lower for forbidden in FORBIDDEN_PATHS):
            return False

        return not (
            path.startswith("/") and not path.startswith(os.path.expanduser("~"))
        )

    def _is_url_safe(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            netloc = parsed.netloc.lower()
            return not any(domain in netloc for domain in FORBIDDEN_DOMAINS)
        except Exception:
            return False

    def _read_file(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"File not found: {path}"}

            if os.path.isdir(path):
                return {"success": False, "error": f"Path is a directory: {path}"}

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "path": path,
                "size": len(content),
            }
        except UnicodeDecodeError:
            return {"success": False, "error": "File contains non-text content"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        try:
            if content is None:
                return {
                    "success": False,
                    "error": "Content cannot be None for write operation",
                }

            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "size": len(content),
                "message": f"Successfully wrote {len(content)} characters to {path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_file_or_directory(
        self, path: str, content: str = None
    ) -> Dict[str, Any]:
        if content is not None:
            return self._write_file(path, content)

        try:
            os.makedirs(path, exist_ok=True)
            return {
                "success": True,
                "path": path,
                "type": "directory",
                "message": f"Created directory: {path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_file_or_directory(self, path: str) -> Dict[str, Any]:
        try:
            if os.path.isfile(path):
                os.remove(path)
                return {
                    "success": True,
                    "path": path,
                    "type": "file",
                    "message": f"Deleted file: {path}",
                }

            if os.path.isdir(path):
                os.rmdir(path)
                return {
                    "success": True,
                    "path": path,
                    "type": "directory",
                    "message": f"Deleted directory: {path}",
                }

            return {"success": False, "error": f"Path not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_directory(self, path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {"success": False, "error": f"Directory not found: {path}"}

            if not os.path.isdir(path):
                return {"success": False, "error": f"Path is not a directory: {path}"}

            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                is_file = os.path.isfile(item_path)
                items.append(
                    {
                        "name": item,
                        "path": item_path,
                        "type": "file" if is_file else "directory",
                        "size": os.path.getsize(item_path) if is_file else None,
                    }
                )

            return {"success": True, "path": path, "items": items, "count": len(items)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_exists(self, path: str) -> Dict[str, Any]:
        try:
            result = {"success": True, "path": path, "exists": os.path.exists(path)}

            if result["exists"]:
                is_file = os.path.isfile(path)
                result.update(
                    {
                        "is_file": is_file,
                        "is_directory": os.path.isdir(path),
                        "size": os.path.getsize(path) if is_file else None,
                    }
                )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _title_of(soup) -> str:
        return soup.title.string if soup.title else "No title"

    def _extract_text_content(self, soup, url: str) -> Dict[str, Any]:
        title = self._title_of(soup)

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return {
            "success": True,
            "url": url,
            "content_type": "text",
            "text": text[:5000],
            "full_length": len(text),
            "title": title,
        }

    def _extract_links(self, soup, url: str) -> Dict[str, Any]:
        links = [
            {
                "text": link.get_text(strip=True),
                "href": link["href"],
                "absolute_url": urljoin(url, link["href"]),
            }
            for link in soup.find_all("a", href=True)
        ]

        return {
            "success": True,
            "url": url,
            "content_type": "links",
            "links": links[:50],
            "total_links": len(links),
        }

    def _extract_images(self, soup, url: str) -> Dict[str, Any]:
        images = [
            {
                "src": img["src"],
                "absolute_url": urljoin(url, img["src"]),
                "alt": img.get("alt", ""),
                "title": img.get("title", ""),
            }
            for img in soup.find_all("img", src=True)
        ]

        return {
            "success": True,
            "url": url,
            "content_type": "images",
            "images": images[:20],
            "total_images": len(images),
        }

    def _extract_metadata(self, soup, url: str) -> Dict[str, Any]:
        metadata = {
            "title": self._title_of(soup),
            "description": "",
            "keywords": "",
            "author": "",
            "og_data": {},
            "twitter_data": {},
        }

        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            property_attr = meta.get("property", "").lower()
            content = meta.get("content", "")

            if name in ("description", "keywords", "author"):
                metadata[name] = content
            elif property_attr.startswith("og:"):
                metadata["og_data"][property_attr] = content
            elif name.startswith("twitter:"):
                metadata["twitter_data"][name] = content

        return {
            "success": True,
            "url": url,
            "content_type": "metadata",
            "metadata": metadata,
        }

    def _extract_all_content(self, soup, url: str) -> Dict[str, Any]:
        # Order matters: text extraction strips <script>/<style> from the tree,
        # so gather links, images and metadata before it runs.
        links_result = self._extract_links(soup, url)
        images_result = self._extract_images(soup, url)
        metadata_result = self._extract_metadata(soup, url)
        text_result = self._extract_text_content(soup, url)

        return {
            "success": True,
            "url": url,
            "content_type": "all",
            "text": text_result["text"],
            "title": text_result["title"],
            "links": links_result["links"][:10],
            "images": images_result["images"][:5],
            "metadata": metadata_result["metadata"],
            "summary": {
                "text_length": text_result["full_length"],
                "total_links": links_result["total_links"],
                "total_images": images_result["total_images"],
            },
        }
