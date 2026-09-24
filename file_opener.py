"""Finds a file (or folder) by its spoken name and opens it with its default
program, as double-clicking it would."""

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from config import FILE_SEARCH_DIRS

SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "site-packages",
    "venv",
    "AppData",
    "Library",
}
MAX_DEPTH = 6
# The search walks real folders (Google Drive streams its files), so it is
# capped rather than allowed to stall the answer.
SEARCH_SECONDS = 5
MAX_CANDIDATES = 8
# Spoken names lose separators: "alfred tray" must find alfred_tray.py.
SEPARATORS = re.compile(r"[\s_\-.]+")

# A default "open" command that runs the file rather than showing it: opening a
# .js or .bat on Windows executes it.
RUNS_THE_FILE = re.compile(
    r"python|\bpyw?\.exe|wscript|cscript|cmd\.exe|powershell|pwsh|mshta|msiexec|"
    r"regedit|javaw?\.exe|^\"?%1",
    re.IGNORECASE,
)
# Outside Windows the association can't be read as easily; these run when opened.
EXECUTABLE_EXTENSIONS = {
    ".app",
    ".bat",
    ".cmd",
    ".command",
    ".exe",
    ".jar",
    ".js",
    ".msi",
    ".ps1",
    ".py",
    ".sh",
    ".vbs",
}


def _squash(text: str) -> str:
    return SEPARATORS.sub("", text.lower())


def _score(query: str, name: str) -> int:
    """3: the exact name, 2: the name without extension, 1: part of the name."""
    squashed_name = _squash(name)
    if query == squashed_name:
        return 3
    if query == _squash(os.path.splitext(name)[0]):
        return 2
    return 1 if query in squashed_name else 0


def find(name: str) -> List[str]:
    """Paths under FILE_SEARCH_DIRS best matching `name`, newest first."""
    query = _squash(os.path.basename(name))
    if not query:
        return []
    best, matches = 0, []
    deadline = time.time() + SEARCH_SECONDS
    for root_dir in FILE_SEARCH_DIRS:
        root_depth = root_dir.rstrip(os.sep).count(os.sep)
        for folder, dirs, files in os.walk(root_dir):
            if time.time() > deadline:
                break
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in SKIP_DIRS
                and folder.count(os.sep) - root_depth < MAX_DEPTH
            ]
            for entry in dirs + files:
                score = _score(query, entry)
                if score and score >= best:
                    if score > best:
                        best, matches = score, []
                    matches.append(os.path.join(folder, entry))
    return sorted(set(matches), key=os.path.getmtime, reverse=True)


def _windows_open_command(path: str) -> str:
    import winreg

    extension = os.path.splitext(path)[1].lower()
    prog_id = None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            rf"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\{extension}\UserChoice",
        ) as key:
            prog_id = winreg.QueryValueEx(key, "ProgId")[0]
    except OSError:
        pass
    try:
        prog_id = prog_id or winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, extension)
        return winreg.QueryValue(
            winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command"
        )
    except OSError:
        return ""


def _would_run(path: str) -> bool:
    if os.path.isdir(path):
        return False
    if sys.platform == "win32":
        return bool(RUNS_THE_FILE.search(_windows_open_command(path)))
    return os.path.splitext(path)[1].lower() in EXECUTABLE_EXTENSIONS


def _is_text(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\x00" not in f.read(4096)


def _launch(command: List[str]):
    subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True
    )


def open_path(path: str) -> Dict[str, Any]:
    # Scripts are opened to be read, not run: in VS Code if there is one.
    # Programs aren't launched at all.
    if _would_run(path):
        editor = shutil.which("code")
        if not _is_text(path):
            return {"error": f"{path} is a program; opening it would run it."}
        if not editor:
            return {"error": f"Opening {path} would run it, and no editor was found."}
        _launch([editor, path])
        return {"opened": path, "with": "VS Code (its default action would run it)"}
    if sys.platform == "win32":
        os.startfile(path)
    else:
        _launch(["open" if sys.platform == "darwin" else "xdg-open", path])
    return {"opened": path}


def open_file(name: str) -> Dict[str, Any]:
    # Only a full path is taken as is: a bare name would resolve against
    # whatever folder AlfreD happens to run from.
    path = os.path.expanduser(name.strip().strip('"'))
    if os.path.isabs(path) and os.path.exists(path):
        return open_path(path)
    matches = find(name)
    if not matches:
        return {
            "error": f"No file or folder matching '{name}' found.",
            "searched": FILE_SEARCH_DIRS,
        }
    if len(matches) > 1:
        return {
            "several_matches": matches[:MAX_CANDIDATES],
            "note": "Ask the user which one, then call open_file with its full path.",
        }
    return open_path(matches[0])
