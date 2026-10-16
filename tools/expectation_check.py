#!/usr/bin/env python3
"""H9 guard: every Expectation markdown cell of nb/p1..p5_*.py must equal its text at the first commit that
contained it (so it was committed before the phase ran and never edited afterwards). Usage: expectation_check.py"""
import glob
import re
import subprocess
import sys


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout


def expectation_cells(text):
    """{heading line: full cell text} for markdown cells whose heading contains 'Expectation'."""
    cells = {}
    for chunk in re.split(r"^(?=# %%)", text, flags=re.M):
        lines = chunk.splitlines()
        if lines and lines[0].startswith("# %% [markdown]") and len(lines) > 1 and "Expectation" in lines[1]:
            cells[lines[1].strip()] = "\n".join(line.rstrip() for line in lines[1:]).strip()
    return cells


def first_versions(path):
    """{heading: cell text at the oldest commit of `path` that contains that heading}."""
    first = {}
    for commit in git("log", "--reverse", "--format=%H", "--", path).split():
        for heading, cell in expectation_cells(git("show", f"{commit}:{path}")).items():
            first.setdefault(heading, cell)
    return first


def main():
    bad = 0
    for path in sorted(glob.glob("nb/p[1-5]_*.py")):
        now = expectation_cells(open(path, encoding="utf-8").read())
        first = first_versions(path.replace("\\", "/"))
        for heading, cell in now.items():
            if heading in first and first[heading] != cell:
                print(f"expectation_check: {path}: {heading!r} differs from its first commit")
                bad += 1
        print(f"expectation_check: {path}: {len(now)} Expectation cell(s) checked")
    sys.exit(1 if bad else 0)


main()
