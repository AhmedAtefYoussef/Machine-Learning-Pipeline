#!/usr/bin/env python3
"""Keep every number in the report traceable to an artifact.

render: fill {{p1.val_r2:.4f}} / {{p4.methods.l1.lambda:.3g}} placeholders from artifacts/*.json
   number_trace.py render report/report.template.md report/report.md [--dir artifacts]
check:  every decimal / percentage number in a text file must match some numeric value in the artifacts
        (relative tol 5e-3 or absolute 5e-4 after rounding), or be whitelisted in report/number_whitelist.txt.
   number_trace.py check report/report.md [--dir artifacts] [--whitelist report/number_whitelist.txt]
Integers < 100 and years 2011/2012 are ignored (page counts, phase numbers, degrees are covered by placeholders when they matter).
"""
import argparse, json, pathlib, re, sys


def load(d):
    return {f.stem: json.loads(f.read_text()) for f in sorted(pathlib.Path(d).glob("*.json"))}


def lookup(arts, path):
    cur = arts
    for k in path.split("."):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


def flat_numbers(o, out):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        out.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            flat_numbers(v, out)
    elif isinstance(o, list):
        for v in o:
            flat_numbers(v, out)


def render(src, dst, d):
    arts = load(d)
    miss = []

    def sub(m):
        key, _, fmt = m.group(1).partition(":")
        try:
            v = lookup(arts, key.strip())
        except (KeyError, IndexError, ValueError, TypeError):
            miss.append(key)
            return m.group(0)
        return format(v, fmt) if fmt else str(v)

    text = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", sub, pathlib.Path(src).read_text())
    pathlib.Path(dst).write_text(text)
    for k in miss:
        print("UNRESOLVED", k)
    print(f"render: {len(miss)} unresolved placeholders")
    sys.exit(1 if miss else 0)


def check(src, d, wl):
    arts = load(d)
    vals = []
    flat_numbers(arts, vals)
    white = set()
    if pathlib.Path(wl).exists():
        white = {l.strip() for l in pathlib.Path(wl).read_text().splitlines() if l.strip() and not l.startswith("#")}
    text = pathlib.Path(src).read_text()
    bad = []
    for m in re.finditer(r"(?<![\w.])(-?\d+\.\d+|\d+(?:\.\d+)?%)", text):
        tok = m.group(1)
        if tok in white:
            continue
        x = float(tok.rstrip("%")) / (100.0 if tok.endswith("%") else 1.0)
        ok = any(abs(x - v) <= max(5e-4, 5e-3 * abs(v)) or abs(x - round(v, 4)) < 1e-9 or (tok.endswith("%") and abs(x * 100 - v) <= 0.6) for v in vals)
        if not ok and not re.fullmatch(r"20(11|12)", tok):
            bad.append(tok)
    for t in sorted(set(bad)):
        print("UNTRACED", t)
    print(f"number_trace: {len(set(bad))} untraced numbers")
    sys.exit(1 if bad else 0)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
r = sp.add_parser("render"); r.add_argument("src"); r.add_argument("dst"); r.add_argument("--dir", default="artifacts")
c = sp.add_parser("check"); c.add_argument("src"); c.add_argument("--dir", default="artifacts"); c.add_argument("--whitelist", default="report/number_whitelist.txt")
a = ap.parse_args()
render(a.src, a.dst, a.dir) if a.cmd == "render" else check(a.src, a.dir, a.whitelist)
