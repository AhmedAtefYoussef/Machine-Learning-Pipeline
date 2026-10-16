"""Make-substitute for the Rush Hour pipeline (stdlib only): `python run.py <target>`.

Targets: kit-check | verify-fast | verify | all | p1..p5 | nb | report | fingerprint | submission | manifest | clean
Same semantics as the Makefile, but it runs on Windows (no make) and uses `python -X utf8` with the environment
from ARCHITECTURE section 0. A phase rebuilds when its artifact is missing, was built from another config or
upstream artifact, or when the content hash of a file it imports differs from artifacts/build_manifest.json.
"""
import ast
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

# stage -> (entry module, upstream artifact, extra input files). A stage's source dependencies are the transitive
# closure of the `src.*` imports of its entry module (see source_deps), so no hand-kept list can go out of date.
STAGES = {
    "p1": ("src/phases/p1.py", None, ["data/train.csv"]),
    "p2": ("src/phases/p2.py", "p1", []),
    "p3": ("src/phases/p3.py", "p2", []),
    "p4": ("src/phases/p4.py", "p3", []),
    "p5": ("src/phases/p5.py", "p4", []),
    "p6": ("src/predict.py", "p5", []),
    "bonus_control": ("src/bonus_control.py", "p1", []),
}
MANIFEST = f"{ART}/build_manifest.json"

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


def sha256_file(path: str) -> str:
    with open(os.path.join(ROOT, path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def content_hash(path: str) -> str:
    """sha256 of a file with CRLF folded to LF, so a Windows and a Linux checkout of one commit agree."""
    with open(os.path.join(ROOT, path), "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def imported_files(path: str) -> set:
    """Repo files (src/...) named by the `import src.x` / `from src.x import y` statements of one module."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names += [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]  # y may be a module
    files = {name.replace(".", "/") + ".py" for name in names if name.startswith("src.")}
    return {f for f in files if os.path.exists(os.path.join(ROOT, f))}


def source_deps(entry: str) -> list:
    """The entry module plus every src/ file it imports, directly or through other src/ modules (sorted)."""
    seen, todo = set(), [entry]
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo += sorted(imported_files(path))
    return sorted(seen)


def stage_inputs(name: str) -> list:
    entry, _, extra = STAGES[name]
    return source_deps(entry) + extra


def input_hashes(name: str) -> dict:
    return {path: content_hash(path) for path in stage_inputs(name)}


def read_manifest() -> dict:
    path = os.path.join(ROOT, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def record(names: list) -> int:
    """Store the current input hashes of the given stages in the manifest."""
    manifest = read_manifest()
    manifest.update({name: input_hashes(name) for name in names})
    with open(os.path.join(ROOT, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return 0


def stale(name: str) -> bool:
    """True when the artifact is missing, was built from another config.yaml or another upstream artifact
    (artifact hashes, so re-running an upstream phase to the same bytes changes nothing), or when the manifest has
    no entry for it or any input file has another content hash than at the last successful build."""
    target = f"{ART}/{name}.json"
    if not os.path.exists(os.path.join(ROOT, target)):
        return True
    with open(os.path.join(ROOT, target), encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("config_sha256") != sha256_file("config.yaml"):
        return True
    upstream = STAGES[name][1]
    if upstream and artifact.get("upstream_sha256") != sha256_file(f"{ART}/{upstream}.json"):
        return True
    return read_manifest().get(name) != input_hashes(name)


def build_stage(name: str, module: str) -> int:
    """Run `python -m module` when the stage is stale; record its input hashes after a clean exit."""
    if not stale(name):
        print(f"[run] {name} up to date")
        return 0
    print(f"[run] building {name}")
    code = run(PY + ["-m", module], tail=30)
    if code != 0:
        print(f"[run] {name} failed (exit {code})")
        return code
    return record([name])


def build_phase(n: int) -> int:
    """Build phases 1..n in order, rebuilding only stale ones (a rebuild makes everything downstream stale)."""
    for k in range(1, n + 1):
        code = build_stage(f"p{k}", f"src.phases.p{k}")
        if code != 0:
            return code
    return 0


def submission() -> int:
    """Build p1..p5 if stale, then the p6 prediction (sample_submission.csv) if stale."""
    return build_phase(5) or build_stage("p6", "src.predict")


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
    run(PY + ["-m", "ruff", "check", "run.py", "src", "tools", "tests"])
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
    if tool("expectation_check") != 0:
        failed.append("expectation_check")
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
    """Bring p1..p5 and the bonus control up to date first, so the rendered placeholders match the artifacts."""
    if build_phase(5) != 0 or build_stage("bonus_control", "src.bonus_control") != 0:
        return 1
    if render_notebook_sources() != 0:
        return 1
    if tool("embed_bundle", "build/nb/p0_setup.py") != 0:   # project files travel inside the notebook (Colab)
        return 1
    code = tool("build_nb", "--src", "build/nb", "--execute", "--out", "rush_hour.ipynb", "--timeout", "3600")
    return code or tool("polish_nb", "rush_hour.ipynb")   # presentation metadata, only after a clean build


def report() -> int:
    code = tool("number_trace", "render", "report/report.template.md", "report/report.md", tail=5)
    if code == 0 and os.path.exists(os.path.join(ROOT, "report", "build_pdf.py")):
        code = run(PY + ["report/build_pdf.py"], tail=5)
    if code == 0:  # the submitted copy at the repo root must match the fresh build
        shutil.copyfile(os.path.join(ROOT, "report", "report.pdf"), os.path.join(ROOT, "report.pdf"))
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
        "submission": submission,
        "manifest": lambda: record(list(STAGES)),
        "clean": clean,
    }
    if target not in simple:
        print(f"unknown target {target!r}")
        return 2
    return simple[target]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
