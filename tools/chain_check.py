#!/usr/bin/env python3
"""Verify the five-phase artifact chain. Exit 1 on any failure.

Artifacts: artifacts/p1.json ... p5.json. Each must carry:
  common:  seed, upstream_sha256 (sha256 of the upstream artifact FILE bytes; null for p1), config_sha256
  p1: weights, feature_names, scaler{mean,std}, lr, iterations, stop_reason, train_loss_final, val_r2, val_rmse, target_transform
  p2: init_loss, init_weights_source ("p1"), degree, feature_names, weights, val_r2, val_rmse
  p3: diagnosis, target_complexity, estimates{seeded,day_holdout,chrono}
  p4: methods{l1,l2,enet:{lambda,[l1_ratio],val_r2}}, survivors_original[], survivors_expanded[], column_verdicts{col:{verdict,evidence}}
  p5: features[], threshold_rule, metrics{accuracy,f1,roc_auc}
Checks: hash chain, seed recompute from config ids, P2 init loss == P1 final loss (tol 1e-9), P5 features subset of P4 survivors,
        P2 starts from P1 weights, every original column has a verdict.
Usage: chain_check.py [--dir artifacts] [--config config.yaml] [--require p1,p2,...]
"""
import argparse, hashlib, json, pathlib, sys

ORIG_COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "dteday", "instant"]
REQ = {
    "p1": ["weights", "feature_names", "scaler", "lr", "iterations", "stop_reason", "train_loss_final", "val_r2", "val_rmse", "target_transform"],
    "p2": ["init_loss", "init_weights_source", "degree", "feature_names", "weights", "val_r2", "val_rmse"],
    "p3": ["diagnosis", "target_complexity", "estimates"],
    "p4": ["methods", "survivors_original", "survivors_expanded", "column_verdicts"],
    "p5": ["features", "threshold_rule", "metrics"],
}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def seed_from_ids(ids):
    return int(hashlib.sha256("_".join(sorted(map(str, ids))).encode()).hexdigest(), 16) % 100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--require", default="p1,p2,p3,p4,p5")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    fails, notes = [], []
    arts = {}
    need = [x for x in a.require.split(",") if x]
    for p in need:
        f = d / f"{p}.json"
        if not f.exists():
            fails.append(f"{p}: missing {f}")
            continue
        arts[p] = json.loads(f.read_text())
        for k in REQ[p] + ["seed", "upstream_sha256", "config_sha256"]:
            if k not in arts[p]:
                fails.append(f"{p}: missing key {k}")
    order = ["p1", "p2", "p3", "p4", "p5"]
    for i, p in enumerate(order[1:], 1):
        if p in arts and order[i - 1] in arts:
            if arts[p].get("upstream_sha256") != sha(d / f"{order[i-1]}.json"):
                fails.append(f"{p}: upstream_sha256 does not match {order[i-1]}.json (stale artifact or hand-edited upstream)")
    seeds = {arts[p].get("seed") for p in arts}
    if len(seeds) > 1:
        fails.append(f"seed differs across artifacts: {seeds}")
    cfg = pathlib.Path(a.config)
    if cfg.exists() and seeds:
        import yaml
        ids = yaml.safe_load(cfg.read_text()).get("team_ids", [])
        if not ids or any("<" in str(x) for x in ids):
            fails.append("config.team_ids still a placeholder")
        elif seed_from_ids(ids) not in seeds:
            fails.append(f"seed in artifacts != recomputed seed {seed_from_ids(ids)}")
    if "p1" in arts and "p2" in arts:
        if abs(arts["p2"]["init_loss"] - arts["p1"]["train_loss_final"]) > 1e-9 * max(1.0, abs(arts["p1"]["train_loss_final"])):
            fails.append(f"P2 init_loss {arts['p2']['init_loss']} != P1 train_loss_final {arts['p1']['train_loss_final']}")
        if arts["p2"].get("init_weights_source") != "p1":
            fails.append("P2 init_weights_source must be 'p1'")
        if arts["p2"]["val_r2"] + 1e-12 < arts["p1"]["val_r2"]:
            notes.append(f"WARN P2 val_r2 {arts['p2']['val_r2']:.4f} < P1 {arts['p1']['val_r2']:.4f}")
    if "p4" in arts:
        v = arts["p4"].get("column_verdicts", {})
        miss = [c for c in ORIG_COLS if c not in v]
        if miss:
            fails.append(f"P4 missing column verdicts: {miss}")
        for c, e in v.items():
            if e.get("verdict") not in {"useful", "redundant", "uninformative"} or not e.get("evidence"):
                fails.append(f"P4 verdict for {c} invalid or lacks evidence")
        if set(arts["p4"].get("methods", {})) != {"l1", "l2", "enet"}:
            fails.append("P4 must contain exactly l1, l2, enet")
    if "p4" in arts and "p5" in arts:
        extra = set(arts["p5"]["features"]) - set(arts["p4"]["survivors_expanded"]) - set(arts["p4"]["survivors_original"])
        if extra:
            fails.append(f"P5 uses features not surviving P4: {sorted(extra)[:10]}")
    for n in notes:
        print(n)
    for f in fails:
        print("FAIL", f)
    print(f"chain_check: {len(arts)}/{len(need)} artifacts, {len(fails)} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
