import json
import logging
import shutil
import subprocess
from typing import List

from tools import Tool, params

logger = logging.getLogger(__name__)

GH_TIMEOUT = 20


def _gh(*args) -> object:
    """Run a read-only gh command and return its parsed JSON output."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GH_TIMEOUT,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"gh exited {result.returncode}"}
    return json.loads(result.stdout or "null")


def github_tools() -> List[Tool]:
    """Read-only GitHub tools via the gh CLI and its existing login. Empty when
    gh isn't installed or signed in, so AlfreD runs fine without it."""
    if not shutil.which("gh"):
        logger.info("GitHub tools disabled: gh CLI not found")
        return []
    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        logger.info("GitHub tools disabled: gh is not logged in (run `gh auth login`)")
        return []

    def notifications(limit: int = 10):
        items = _gh("api", "notifications", "--jq", f".[:{int(limit)}]")
        if isinstance(items, dict):
            return items
        return [
            {
                "repo": n["repository"]["full_name"],
                "type": n["subject"]["type"],
                "title": n["subject"]["title"],
                "reason": n["reason"],
                "updated": n["updated_at"],
            }
            for n in items
        ] or "No unread notifications."

    def my_pull_requests(state: str = "open"):
        return (
            _gh(
                "search",
                "prs",
                "--author=@me",
                f"--state={state}",
                "--limit",
                "10",
                "--json",
                "title,repository,state,updatedAt",
            )
            or f"No {state} pull requests."
        )

    def my_issues(state: str = "open"):
        return (
            _gh(
                "search",
                "issues",
                "--assignee=@me",
                f"--state={state}",
                "--limit",
                "10",
                "--json",
                "title,repository,state,updatedAt",
            )
            or f"No {state} issues assigned to you."
        )

    def recent_repos(limit: int = 5):
        return _gh(
            "repo",
            "list",
            "--limit",
            str(int(limit)),
            "--json",
            "name,description,pushedAt,stargazerCount",
        )

    def repo_activity(repo: str):
        return {
            "commits": _gh(
                "api",
                f"repos/{repo}/commits?per_page=5",
                "--jq",
                "[.[] | {message: .commit.message, author: .commit.author.name, date: .commit.author.date}]",
            ),
            "open_prs": _gh(
                "pr", "list", "--repo", repo, "--json", "title,author,updatedAt"
            ),
            "open_issues": _gh(
                "issue", "list", "--repo", repo, "--json", "title,author,updatedAt"
            ),
        }

    state = {"type": "string", "enum": ["open", "closed"]}
    return [
        Tool(
            "github_notifications",
            "List the user's unread GitHub notifications.",
            params(limit={"type": "integer"}),
            notifications,
        ),
        Tool(
            "github_my_pull_requests",
            "List pull requests authored by the user.",
            params(state=state),
            my_pull_requests,
        ),
        Tool(
            "github_my_issues",
            "List GitHub issues assigned to the user.",
            params(state=state),
            my_issues,
        ),
        Tool(
            "github_recent_repos",
            "List the user's most recently pushed repositories.",
            params(limit={"type": "integer"}),
            recent_repos,
        ),
        Tool(
            "github_repo_activity",
            "Recent commits, open PRs and open issues for a repository.",
            params(["repo"], repo={"type": "string", "description": "owner/name"}),
            repo_activity,
        ),
    ]
