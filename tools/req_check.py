#!/usr/bin/env python3
"""Requirements traceability gate. Exit 1 if any requirement lacks code/test/artifact refs or is not 'verified' when --strict.
docs/TRACE.csv columns: req_id,statement,phase,code_ref,test_ref,artifact_ref,status,verifier_note
status in: todo | implemented | verified | waived(<reason in verifier_note>)
Usage: req_check.py [--trace docs/TRACE.csv] [--strict] [--phase p3]   (without --strict: only reports counts; fails on missing refs for implemented/verified rows)
"""
import argparse, csv, collections, pathlib, sys

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default="docs/TRACE.csv")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--phase")
a = ap.parse_args()
rows = list(csv.DictReader(open(a.trace)))
if a.phase:
    rows = [r for r in rows if r["phase"] in (a.phase, "all")]
c = collections.Counter(r["status"].split("(")[0] for r in rows)
fails = []
for r in rows:
    st = r["status"].split("(")[0]
    if st in {"implemented", "verified"}:
        for k in ("code_ref", "test_ref"):
            if not r[k].strip():
                fails.append(f"{r['req_id']}: status={st} but {k} empty")
        if st == "verified" and not r["artifact_ref"].strip():
            fails.append(f"{r['req_id']}: verified without artifact_ref")
    if st == "waived" and not r["verifier_note"].strip():
        fails.append(f"{r['req_id']}: waived without reason")
    if a.strict and st != "verified" and st != "waived":
        fails.append(f"{r['req_id']}: not verified ({r['status']})")
for f in fails:
    print("FAIL", f)
print(f"req_check: {len(rows)} reqs {dict(c)}; {len(fails)} failures")
sys.exit(1 if fails else 0)
