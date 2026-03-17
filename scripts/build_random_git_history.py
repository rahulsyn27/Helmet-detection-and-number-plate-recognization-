#!/usr/bin/env python3
"""
Build backdated git history: March 2–21 (one year), random 1–2 days skipped in the middle,
1–5 commits per active day, random times. Splits all currently-untracked (non-ignored) files
across commits in path order.

Requires: clean repo with NO commits yet, git user.name / user.email configured.
Re-run with same GIT_HISTORY_SEED to reproduce the same calendar.

Usage (from repo root):
  python3 scripts/build_random_git_history.py

If you already have commits, reset first (keeps files on disk):

  git update-ref -d HEAD
  git rm -rf --cached .
  python3 scripts/build_random_git_history.py
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# March window (inclusive). Change YEAR if you want another March.
YEAR = 2026
START_DAY = 2
END_DAY = 21
SKIP_DAYS_MIN = 1
SKIP_DAYS_MAX = 2


def run_git(args: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(["git", *args], cwd=str(REPO_ROOT), check=True, env=env)


def git_output(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def ensure_fresh_repo() -> None:
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    if r.returncode == 0:
        print(
            "This repo already has commits. To rebuild history (files stay on disk), run:\n"
            "  git update-ref -d HEAD\n"
            "  git rm -rf --cached .\n"
            "  python3 scripts/build_random_git_history.py",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_identity() -> None:
    name = git_output(["config", "user.name"]).strip()
    email = git_output(["config", "user.email"]).strip()
    if not name or not email:
        print(
            "Set git user.name and user.email first, e.g.\n"
            '  git config user.name "Your Name"\n'
            '  git config user.email "you@example.com"',
            file=sys.stderr,
        )
        sys.exit(1)


def list_files_to_commit() -> list[str]:
    """Paths to include: staged/cached plus untracked (not ignored)."""
    cached = git_output(["ls-files"]).splitlines()
    other = git_output(["ls-files", "-o", "--exclude-standard"]).splitlines()
    return sorted({p for p in cached + other if p})


def build_active_days() -> tuple[list[int], list[int]]:
    all_days = list(range(START_DAY, END_DAY + 1))
    skip_n = random.randint(SKIP_DAYS_MIN, SKIP_DAYS_MAX)
    # Skip only from the middle (not first/last day) so the range still "reads" March 2–21.
    candidates = [d for d in all_days if d not in (START_DAY, END_DAY)]
    if not candidates:
        return all_days, []
    skip_n = min(skip_n, len(candidates))
    skipped = sorted(random.sample(candidates, skip_n))
    active = [d for d in all_days if d not in set(skipped)]
    return active, skipped


def random_time_parts() -> tuple[int, int, int]:
    return random.randint(8, 22), random.randint(0, 59), random.randint(0, 59)


def build_schedule(active_days: list[int]) -> list[tuple[int, int, int, int]]:
    """List of (day, hour, minute, second), sorted chronologically."""
    slots: list[tuple[int, int, int, int]] = []
    for day in active_days:
        n_commits = random.randint(1, 5)
        for _ in range(n_commits):
            h, m, s = random_time_parts()
            slots.append((day, h, m, s))
    slots.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return slots


def format_git_date(year: int, day: int, hour: int, minute: int, second: int) -> str:
    # Git accepts: YYYY-MM-DD HH:MM:SS (interpreted in local TZ)
    return f"{year}-03-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def split_files_into_commits(files: list[str], n_commits: int) -> list[list[str]]:
    """Split files into n_commits chunks; if n_commits > len(files), trailing commits are empty."""
    if n_commits <= 0:
        return []
    if not files:
        return [[] for _ in range(n_commits)]
    if n_commits >= len(files):
        chunks = [[f] for f in files]
        while len(chunks) < n_commits:
            chunks.append([])
        return chunks
    chunks: list[list[str]] = []
    n = len(files)
    base, rem = divmod(n, n_commits)
    idx = 0
    for i in range(n_commits):
        take = base + (1 if i < rem else 0)
        chunks.append(files[idx : idx + take])
        idx += take
    return chunks


def main() -> None:
    ensure_fresh_repo()
    ensure_identity()

    seed_str = os.environ.get("GIT_HISTORY_SEED")
    if seed_str is not None:
        seed = int(seed_str)
    else:
        seed = random.randrange(1 << 31)
    random.seed(seed)
    print(f"GIT_HISTORY_SEED={seed}  (export to reproduce this exact schedule)")

    active_days, skipped = build_active_days()
    print(f"Active days (March): {active_days}")
    print(f"Skipped days: {skipped}")

    schedule = build_schedule(active_days)
    n_commits = len(schedule)
    print(f"Total commits: {n_commits}")

    files = list_files_to_commit()
    if not files:
        print("No untracked files (after .gitignore). Nothing to commit.", file=sys.stderr)
        sys.exit(1)

    chunks = split_files_into_commits(files, n_commits)
    assert len(chunks) == n_commits

    for i, ((day, hour, minute, second), batch) in enumerate(zip(schedule, chunks), start=1):
        date_str = format_git_date(YEAR, day, hour, minute, second)
        extra_env = {
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_DATE": date_str,
        }
        if batch:
            run_git(["add", "--"] + batch)
        msg = f"checkpoint {i}/{n_commits} (Mar {day})"
        if batch:
            run_git(["commit", "-m", msg], extra_env=extra_env)
        else:
            run_git(["commit", "--allow-empty", "-m", msg], extra_env=extra_env)

    print("Done. Verify with: git log --oneline --date=short --format='%h %ad %s'")


if __name__ == "__main__":
    main()
