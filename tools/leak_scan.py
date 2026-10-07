#!/usr/bin/env python3
"""AST scan for data-leakage and forbidden-API violations. Exit 1 on any finding.

Rules
  L1  .fit/.fit_transform/.partial_fit called with an argument whose name looks like val/valid/test
  L2  test.csv read outside the allowed final-stage files (config: --test-ok)
  L3  P1/P2 files (matched by --scratch-glob) call sklearn estimators' fit/predict, or import sklearn.linear_model
Suppress a line with the comment  `# leak-ok: <reason>`  (use for oracle tests only; reason is printed).

Usage: leak_scan.py [paths...] [--scratch-glob 'nb/p1*.py' --scratch-glob 'nb/p2*.py' --scratch-glob 'src/gd*.py'] [--test-ok src/predict.py nb/p6*.py]
"""
import argparse, ast, fnmatch, pathlib, re, sys

SUSP = re.compile(r"(^|_)(val|valid|validation|test|te|xte|xval|x_val|x_te|yval|y_val)($|_)|^(xv|xt)$", re.I)
FIT_FUNCS = {"fit", "fit_transform", "partial_fit"}


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def scan(path, scratch_globs, test_ok):
    src = path.read_text()
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [(path, e.lineno or 0, "SYNTAX", str(e))]
    out = []
    scratch = any(fnmatch.fnmatch(str(path), g) for g in scratch_globs)
    ok_test = any(fnmatch.fnmatch(str(path), g) for g in test_ok)

    def suppressed(lineno):
        return 0 < lineno <= len(lines) and "leak-ok" in lines[lineno - 1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func.attr
            if fn in FIT_FUNCS:
                args = list(node.args) + [k.value for k in node.keywords]
                bad = sorted({n for a in args for n in names_in(a) if SUSP.search(n)})
                if bad and not suppressed(node.lineno):
                    out.append((path, node.lineno, "L1", f".{fn}() sees {bad}"))
            if scratch and fn in FIT_FUNCS | {"predict"} and not suppressed(node.lineno):
                recv = names_in(node.func.value)
                if recv & {"model", "mdl", "clf", "reg", "lr", "est", "ridge", "lasso", "sklearn", "LinearRegression", "Ridge", "Lasso", "SGDRegressor"} or fn == "predict":
                    out.append((path, node.lineno, "L3", f"library .{fn}() in from-scratch phase"))
        if isinstance(node, (ast.Import, ast.ImportFrom)) and scratch:
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            if ("sklearn.linear_model" in mod or any("sklearn.linear_model" in n for n in names)) and not suppressed(node.lineno):
                out.append((path, node.lineno, "L3", "sklearn.linear_model imported in from-scratch phase"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"test\.csv", node.value):
            if not ok_test and not suppressed(node.lineno):
                out.append((path, node.lineno, "L2", "test.csv referenced outside allowed final-stage files"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["src", "nb"])
    ap.add_argument("--scratch-glob", action="append", default=[])
    ap.add_argument("--test-ok", nargs="*", default=["*predict*", "*submission*", "*common.py", "*data*.py"])
    a = ap.parse_args()
    files = []
    for p in a.paths:
        p = pathlib.Path(p)
        files += [p] if p.is_file() else sorted(p.rglob("*.py"))
    findings = []
    for f in files:
        findings += scan(f, a.scratch_glob, a.test_ok)
    for f, ln, rule, msg in findings:
        print(f"{rule} {f}:{ln} {msg}")
    print(f"leak_scan: {len(files)} files, {len(findings)} findings")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
