"""Make-substitute for the Rush Hour pipeline (stdlib only): `python run.py <target>`.

Targets: kit-check | verify-fast | verify | all | p1..p5 | nb | report | fingerprint | submission | clean
Same semantics as the Makefile, but it runs on Windows (no make) and uses `python -X utf8` with the environment
from ARCHITECTURE section 0. A phase rebuilds when its artifact is missing or older than an input.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-X", "utf8"]
ART = "artifacts"

# phase -> (upstream artifact, files named in the Makefile dependency list)
PHASE_DEPS = {
    1: (None, ["src/gd.py", "src/phases/p1.py"]),
    2: ("p1", ["src/poly.py", "src/phases/p2.py"]),
    3: ("p2", ["src/validation.py", "src/phases/p3.py"]),
    4: ("p3", ["src/regularization.py", "src/phases/p4.py"]),
    5: ("p4", ["src/logistic.py", "src/phases/p5.py"]),
}
EVERY_PHASE = ["config.yaml", "src/features.py", "src/common.py"]

LEAK_ARGS = ["src", "nb",
             "--scratch-glob", "*p1_*.py", "--scratch-glob", "*p2_*.py",
             "--scratch-glob", "*phases?p1.py", "--scratch-glob", "*phases?p2.py",
             "--scratch-glob", "*gd*.py", "--scratch-glob", "*poly.py",
             "--test-ok", "*predict*", "*submission*", "*common.py"]


def env() -> dict:
    """Process environment with single-threaded BLAS and PYTHONPATH=. (ARCHITECTURE section 0)."""
    e = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        e[var] = "1"
    e["PYTHONPATH"] = "."
    e["PYTHONUTF8"] = "1"
    return e


def run(cmd: list, tail: int | None = 15) -> int:
    """Run a command, print the last `tail` lines of its output (all if None), return its exit code."""
    proc = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = proc.stdout.decode("utf-8", errors="replace").splitlines()
    for line in (lines if tail is None else lines[-tail:]):
        print(line)
    return proc.returncode


def tool(name: str, *args: str, tail: int = 15) -> int:
    """Run tools/<name>.py."""
    return run(PY + [f"tools/{name}.py", *args], tail)


def mtime(path: str) -> float:
    return os.path.getmtime(os.path.join(ROOT, path))


def stale(n: int) -> bool:
    """True when artifacts/pN.json is missing or older than any existing input."""
    target = f"{ART}/p{n}.json"
    if not os.path.exists(os.path.join(ROOT, target)):
        return True
    upstream, own = PHASE_DEPS[n]
    inputs = EVERY_PHASE + own + ([f"{ART}/{upstream}.json"] if upstream else [])
    return any(os.path.exists(os.path.join(ROOT, p)) and mtime(p) > mtime(target) for p in inputs)


def build_phase(n: int) -> int:
    """Build phases 1..n in order, rebuilding only stale ones (a rebuild makes everything downstream stale)."""
    for k in range(1, n + 1):
        if stale(k):
            print(f"[run] building p{k}")
            code = run(PY + ["-m", f"src.phases.p{k}"], tail=30)
            if code != 0:
                print(f"[run] p{k} failed (exit {code})")
                return code
        else:
            print(f"[run] p{k} up to date")
    return 0


def existing_artifacts() -> str:
    """Names of the existing p1..p5 artifacts (chain_check knows only these; p6 has its own check)."""
    return ",".join(f"p{n}" for n in range(1, 6) if os.path.exists(os.path.join(ROOT, ART, f"p{n}.json")))


def p6_chain() -> bool:
    """Print and return whether p6.upstream_sha256 equals the sha256 of artifacts/p5.json (True if no p6 yet)."""
    p6, p5 = (os.path.join(ROOT, ART, f"{n}.json") for n in ("p6", "p5"))
    if not (os.path.exists(p6) and os.path.exists(p5)):
        print("p6 chain: no p6 artifact")
        return True
    with open(p6, "r", encoding="utf-8") as fh:
        recorded = json.load(fh).get("upstream_sha256")
    with open(p5, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    ok = recorded == actual
    print("p6 chain: upstream ok" if ok else "p6 chain: MISMATCH")
    return ok


def verify_fast() -> int:
    """ruff (non-fatal), pytest, leak_scan, chain_check, req_check; fails if pytest, leak_scan or chain_check fail."""
    run(PY + ["-m", "ruff", "check", "src", "tools", "tests"])
    failed = []
    code = run(PY + ["-m", "pytest", "-q", "-x", "tests"])
    if code not in (0, 5):  # 5 = no tests collected yet
        failed.append("pytest")
    if tool("leak_scan", *LEAK_ARGS) != 0:
        failed.append("leak_scan")
    have = existing_artifacts()
    if have:
        if tool("chain_check", "--require", have) != 0:
            failed.append("chain_check")
    else:
        print("chain_check: no artifacts yet")
    if not p6_chain():
        failed.append("p6 chain")
    tool("req_check", tail=10)
    if failed:
        print("verify-fast FAILED:", ", ".join(failed))
        return 1
    return 0


def render_notebook_sources() -> int:
    """Fill {{...}} placeholders of every nb/p*.py into build/nb/."""
    out_dir = os.path.join(ROOT, "build", "nb")
    os.makedirs(out_dir, exist_ok=True)
    bad = 0
    for src in sorted(glob.glob(os.path.join(ROOT, "nb", "p*.py"))):
        name = os.path.basename(src)
        bad |= tool("number_trace", "render", f"nb/{name}", f"build/nb/{name}", tail=5) != 0
    return int(bad)


def nb() -> int:
    if render_notebook_sources() != 0:
        return 1
    return tool("build_nb", "--src", "build/nb", "--execute", "--out", "rush_hour.ipynb", "--timeout", "3600")


def report() -> int:
    code = tool("number_trace", "render", "report/report.template.md", "report/report.md", tail=5)
    if code == 0 and os.path.exists(os.path.join(ROOT, "report", "build_pdf.py")):
        code = run(PY + ["report/build_pdf.py"], tail=5)
    return code


def verify() -> int:
    failed = []
    if verify_fast() != 0:
        failed.append("verify-fast")
    if nb() != 0:
        failed.append("nb")
    if tool("submission_check", tail=10) != 0:
        failed.append("submission_check")
    if tool("number_trace", "render", "report/report.template.md", "report/report.md", tail=5) != 0:
        failed.append("number_trace render")
    if tool("number_trace", "check", "report/report.md") != 0:
        failed.append("number_trace check")
    if tool("req_check", "--strict") != 0:
        failed.append("req_check --strict")
    if failed:
        print("verify FAILED:", ", ".join(failed))
        return 1
    return 0


def kit_check() -> int:
    if not os.path.exists(os.path.join(ROOT, "CLAUDE.md")):
        print("CLAUDE.md missing")
        return 1
    files = sorted(glob.glob(os.path.join(ROOT, ".claude", "agents", "*.md"))
                   + glob.glob(os.path.join(ROOT, ".claude", "skills", "*", "SKILL.md")))
    print("kit files:", len(files))
    missing = []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            if not any(line.startswith("name:") for line in fh):
                missing.append(path)
    print("\n".join(missing) if missing else "frontmatter ok")
    return 1 if missing else 0


def clean() -> int:
    for path in glob.glob(os.path.join(ROOT, ART, "p?.json")) + [os.path.join(ROOT, "rush_hour.ipynb"),
                                                                 os.path.join(ROOT, "report", "report.md")]:
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.join(ROOT, "build", "nb"), ignore_errors=True)
    return 0


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = argv[1]
    if target in ("p1", "p2", "p3", "p4", "p5"):
        return build_phase(int(target[1]))
    simple = {
        "all": lambda: build_phase(5),
        "kit-check": kit_check,
        "verify-fast": verify_fast,
        "verify": verify,
        "nb": nb,
        "report": report,
        "fingerprint": lambda: tool("fingerprint", ART),
        "submission": lambda: run(PY + ["-m", "src.predict"]),
        "clean": clean,
    }
    if target not in simple:
        print(f"unknown target {target!r}")
        return 2
    return simple[target]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
