#!/usr/bin/env python3
"""Print a stable sha256 per artifact (volatile keys removed) so behavior-preserving refactors can be proven:
run before and after, diff the output. Usage: fingerprint.py [artifacts_dir]"""
import hashlib, json, pathlib, sys

VOLATILE = {"created_at", "runtime_s", "git_commit", "host"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in sorted(o.items()) if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, float):
        return round(o, 10)
    return o


d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
for f in sorted(d.glob("p?.json")):
    print(f.stem, hashlib.sha256(json.dumps(strip(json.loads(f.read_text())), sort_keys=True).encode()).hexdigest()[:16])
