#!/usr/bin/env python3
"""Assemble percent-format phase files (nb/p0*.py ... nb/p6*.py) into one notebook and optionally run it top-to-bottom.
Why: notebook JSON merges badly and burns tokens; plain .py cell files diff cleanly and can be owned per phase.
Cell syntax:   `# %%` code cell  |  `# %% [markdown]` markdown cell (following lines are '# ' comments)
No third-party dependency: writes nbformat-4 JSON directly; --execute uses nbclient if installed, else a built-in sequential
executor (shared namespace, stdout captured) that is sufficient to prove 'runs top-to-bottom without errors'.
Usage: build_nb.py [--src nb] [--out rush_hour.ipynb] [--execute] [--max-out-lines 30] [--timeout 1800]
"""
import argparse, contextlib, glob, io, json, pathlib, re, sys, time, traceback


def parse(path):
    cells, cur, kind = [], [], None
    def flush():
        nonlocal cur, kind
        if kind is not None and any(l.strip() for l in cur):
            src = "\n".join(cur).strip("\n")
            if kind == "markdown":
                src = "\n".join(re.sub(r"^# ?", "", l) for l in src.split("\n"))
            cells.append((kind, src, path.name))
        cur = []
    for line in path.read_text().splitlines():
        m = re.match(r"^# %%(\s*\[markdown\])?\s*(.*)$", line)
        if m:
            flush(); kind = "markdown" if m.group(1) else "code"
        else:
            cur.append(line)
    flush()
    return cells


def to_ipynb(cells):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "cells": []}
    for i, (k, s, f) in enumerate(cells):
        c = {"id": f"c{i:04d}", "cell_type": k, "metadata": {"src": f}, "source": s.splitlines(True)}
        if k == "code":
            c.update(outputs=[], execution_count=None)
        nb["cells"].append(c)
    return nb


def execute(nb, max_lines, timeout):
    try:
        import nbclient, nbformat
        for c in nb["cells"]:  # nbclient needs string sources (kit fix: lists crashed it)
            c["source"] = "".join(c["source"])
        nbo = nbformat.from_dict(nb)
        nbclient.NotebookClient(nbo, timeout=timeout, kernel_name="python3").execute()
        return nbformat.to_notebook_dict(nbo) if hasattr(nbformat, "to_notebook_dict") else json.loads(nbformat.writes(nbo)), []
    except ImportError:
        pass
    ns, errs, t0 = {"__name__": "__main__"}, [], time.time()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile("".join(c["source"]), f"<{c['metadata']['src']}:{c['id']}>", "exec"), ns)
        except Exception:
            errs.append((c["metadata"]["src"], c["id"], traceback.format_exc(limit=3)))
            break
        out = buf.getvalue().splitlines()
        if out:
            c["outputs"] = [{"output_type": "stream", "name": "stdout", "text": [l + "\n" for l in out[:max_lines]]}]
        if time.time() - t0 > timeout:
            errs.append(("-", "-", "timeout")); break
    return nb, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="nb"); ap.add_argument("--out", default="rush_hour.ipynb")
    ap.add_argument("--execute", action="store_true"); ap.add_argument("--max-out-lines", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    files = sorted(pathlib.Path(a.src).glob("p[0-9]*.py"))
    if not files:
        print("build_nb: no nb/pN*.py files"); sys.exit(1)
    cells = [c for f in files for c in parse(f)]
    nb, errs = to_ipynb(cells), []
    # structural check: each phase file starts with an Expectation markdown cell and ends with an Outcome markdown cell
    problems = []
    for f in files:
        if re.match(r"p[1-5]", f.name):
            cs = parse(f)
            md = [c for c in cs if c[0] == "markdown"]
            if not md or not re.match(r"\s*#*\s*Expectation", md[0][1], re.I): problems.append(f"{f.name}: first markdown cell must be titled Expectation")
            if not md or not re.match(r"\s*#*\s*Outcome", md[-1][1], re.I): problems.append(f"{f.name}: last markdown cell must be titled Outcome")
            if cs and cs[0][0] != "markdown": problems.append(f"{f.name}: Expectation cell must come first")
    if a.execute:
        nb, errs = execute(nb, a.max_out_lines, a.timeout)
    pathlib.Path(a.out).write_text(json.dumps(nb, indent=1))
    for p in problems: print("STRUCT", p)
    for f, cid, tb in errs: print(f"ERROR {f} {cid}\n{tb}")
    print(f"build_nb: {len(cells)} cells -> {a.out}; executed={a.execute}; errors={len(errs)}; structure_problems={len(problems)}")
    sys.exit(1 if (errs or problems) else 0)


if __name__ == "__main__":
    main()
