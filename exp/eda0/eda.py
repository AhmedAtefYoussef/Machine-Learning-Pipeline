"""EDA0 (chief, diagnostic only): re-measure data facts with the real seed and test the Phase-0 design options.
Uses numpy lstsq as a fast oracle. Nothing here enters the chain. Output: exp/eda0/eda.json + printed digest.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
seed = int(hashlib.sha256("_".join(sorted(cfg["team_ids"])).encode()).hexdigest(), 16) % 100000
tr_all = pd.read_csv("data/train.csv", parse_dates=["dteday"])
te = pd.read_csv("data/test.csv", parse_dates=["dteday"])
out = {"seed": seed}
T0 = pd.Timestamp("2011-01-01")

# ---------- A. facts ----------
f = {}
f["train_shape"], f["test_shape"] = list(tr_all.shape), list(te.shape)
f["nan"] = int(tr_all.isna().sum().sum() + te.isna().sum().sum())
f["train_days"] = int(tr_all.dteday.nunique())
f["train_dom"] = sorted(tr_all.dteday.dt.day.unique().tolist())
f["test_dom"] = sorted(te.dteday.dt.day.unique().tolist())
f["test_days"] = int(te.dteday.nunique())
f["days_lt24"] = int((tr_all.groupby("dteday").size() < 24).sum())
f["rows_per_hr_min"] = tr_all.groupby("hr").size().nsmallest(3).to_dict()
c = tr_all.cnt
f["cnt"] = {k: float(v) for k, v in dict(mean=c.mean(), median=c.median(), sd=c.std(), skew=c.skew(), min=c.min(), max=c.max(),
                                         q75=c.quantile(.75), q90=c.quantile(.9), q99=c.quantile(.99)).items()}
f["yr_ratio"] = float(tr_all[tr_all.yr == 1].cnt.mean() / tr_all[tr_all.yr == 0].cnt.mean())
f["r_temp_atemp"] = float(tr_all.temp.corr(tr_all.atemp))
f["r_season_mnth"] = float(tr_all.season.corr(tr_all.mnth))
f["r_instant_days"] = float(tr_all.instant.corr((tr_all.dteday - T0).dt.days))
f["r_yr_days"] = float(tr_all.yr.corr((tr_all.dteday - T0).dt.days))
wd_calc = ((tr_all.weekday.isin([1, 2, 3, 4, 5])) & (tr_all.holiday == 0)).astype(int)
f["workingday_inconsistent"] = int((wd_calc != tr_all.workingday).sum())
f["weathersit_counts"] = tr_all.weathersit.value_counts().to_dict()
f["test_weathersit_counts"] = te.weathersit.value_counts().to_dict()
f["hum0_rows"] = int((tr_all.hum == 0).sum()); f["hum0_days"] = tr_all[tr_all.hum == 0].dteday.dt.strftime("%Y-%m-%d").unique().tolist()
f["wind0_share_train"] = float((tr_all.windspeed == 0).mean()); f["wind0_share_test"] = float((te.windspeed == 0).mean())
d = (tr_all.atemp - tr_all.temp).abs()
f["atemp_bad_days"] = tr_all[d > 0.2].dteday.dt.strftime("%Y-%m-%d").value_counts().to_dict()
f["holiday_rows"] = int(tr_all.holiday.sum()); f["test_holiday_rows"] = int(te.holiday.sum())
prof = tr_all.groupby(["workingday", "hr"]).cnt.mean().unstack(0)
f["peak_wd"] = prof[1].nlargest(3).round(0).to_dict(); f["peak_nwd"] = prof[0].nlargest(3).round(0).to_dict()
out["facts"] = f

# ---------- B. split ----------
tr, va = train_test_split(tr_all, test_size=0.20, random_state=seed)
out["split"] = {"n_train": len(tr), "n_val": len(va), "train_cnt_mean": float(tr.cnt.mean()), "val_cnt_mean": float(va.cnt.mean()),
                "val_days_also_in_train": float(va.dteday.isin(tr.dteday).mean())}


# ---------- C. design prototype ----------
def base_cols(df, ref, opts):
    """Return DataFrame of unscaled features. ref = training frame used for any fitted statistic."""
    X = {}
    for h in range(1, 24):
        X[f"hr_{h}"] = (df.hr == h).astype(float)
    X["workingday"] = df.workingday.astype(float)
    X["holiday"] = df.holiday.astype(float)
    ws = df.weathersit.clip(upper=3)
    X["ws_2"] = (ws == 2).astype(float); X["ws_3"] = (ws == 3).astype(float)
    hum = df.hum.astype(float).copy()
    if opts.get("hum_fix", True):
        hum[df.hum == 0] = ref.hum[ref.hum > 0].median()
    X["temp"] = df.temp; X["hum"] = hum; X["windspeed"] = df.windspeed
    if opts.get("wind_flag"):
        X["wind_zero"] = (df.windspeed == 0).astype(float)
    days = (df.dteday - T0).dt.days.astype(float)
    X["trend"] = days
    doy = df.dteday.dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if opts.get("doy2"):
        X["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25); X["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    if opts.get("hr_cyclic"):
        for h in range(1, 24):
            del X[f"hr_{h}"]
        for k in range(1, opts["hr_cyclic"] + 1):
            X[f"hr_s{k}"] = np.sin(2 * np.pi * k * df.hr / 24); X[f"hr_c{k}"] = np.cos(2 * np.pi * k * df.hr / 24)
    if opts.get("hr_numeric"):
        for h in range(1, 24):
            del X[f"hr_{h}"]
        X["hr"] = df.hr.astype(float)
    return pd.DataFrame(X, index=df.index)


def expand(B, df, opts):
    X = B.copy()
    if opts.get("wdhr"):
        for h in range(1, 24):
            X[f"wd_x_hr_{h}"] = B["workingday"] * B[f"hr_{h}"]
    return X


def powers(Z, cols, deg, cross=False):
    """Z standardized base frame; add z^2..z^deg for cols."""
    N = {}
    for c_ in cols:
        for p in range(2, deg + 1):
            N[f"{c_}^{p}"] = Z[c_] ** p
    if cross:
        N["temp*hum"] = Z["temp"] * Z["hum"]
    return pd.DataFrame(N, index=Z.index)


def build(trn, others, opts):
    """Fit on trn; return standardized matrices [trn]+others with intercept, names."""
    frames = [expand(base_cols(d_, trn, opts), d_, opts) for d_ in [trn] + others]
    mu, sd = frames[0].mean(), frames[0].std(ddof=0).replace(0, 1)
    Zs = [(F - mu) / sd for F in frames]
    deg = opts.get("deg", 1)
    if deg > 1 or opts.get("cross"):
        Ps = [powers(Z, opts.get("pow_cols", ["temp", "hum", "windspeed"]), deg, opts.get("cross", False)) for Z in Zs]
        m2, s2 = Ps[0].mean(), Ps[0].std(ddof=0).replace(0, 1)
        Zs = [pd.concat([Z, (P - m2) / s2], axis=1) for Z, P in zip(Zs, Ps)]
    names = ["bias"] + list(Zs[0].columns)
    return [np.c_[np.ones(len(Z)), Z.values] for Z in Zs], names


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_eval(trn, vals, opts, target="log", ridge=1e-8):
    Xs, names = build(trn, vals, opts)
    Xt = Xs[0]
    y = np.log1p(trn.cnt.values) if target == "log" else trn.cnt.values.astype(float)
    A = Xt.T @ Xt / len(Xt); A[np.diag_indices_from(A)] += ridge; A[0, 0] -= ridge
    w = np.linalg.solve(A, Xt.T @ y / len(Xt))
    res = []
    eta_t = Xt @ w
    smear = float(np.mean(np.exp(y - eta_t))) if target == "log" else 1.0
    for X, d_ in zip(Xs, [trn] + vals):
        eta = X @ w
        p = np.exp(eta) * smear - 1 if target == "log" else eta
        p = np.clip(p, 0, None)
        res.append(float(r2(d_.cnt.values, p)))
    return res, smear, len(names), (Xt, y, w)


def day_folds(trn, k=5):
    days = np.sort(trn.dteday.unique())
    fold_of = {d_: i % k for i, d_ in enumerate(days)}
    return trn.dteday.map(fold_of).values


def three(opts, target="log"):
    (r_tr, r_va), smear, p, _ = fit_eval(tr, [va], opts, target)
    fo = day_folds(tr)
    cv = [fit_eval(tr[fo != i], [tr[fo == i]], opts, target)[0][1] for i in range(5)]
    cut = pd.Timestamp(cfg["chrono"]["cut_date"])
    (c_tr, c_va), *_ = fit_eval(tr_all[tr_all.dteday < cut], [tr_all[tr_all.dteday >= cut]], opts, target)
    return dict(p=p, train=round(r_tr, 4), seeded=round(r_va, 4), daycv=round(float(np.mean(cv)), 4), daycv_sd=round(float(np.std(cv)), 4),
                chrono_tr=round(c_tr, 4), chrono=round(c_va, 4), smear=round(smear, 4))


V = {}
V["P1base_raw"] = three({}, "raw")
V["P1base_log"] = three({}, "log")
V["P1base_log_nohumfix"] = three({"hum_fix": False})
V["P1base_log_windflag"] = three({"wind_flag": True})
V["P1base_log_doy2"] = three({"doy2": True})
V["P1_hrnumeric_log"] = three({"hr_numeric": True}); V["P1_hrnumeric_raw"] = three({"hr_numeric": True}, "raw")
for k in (2, 4, 6):
    V[f"P1_hrcyclic{k}_log"] = three({"hr_cyclic": k})
V["P2_wdhr_raw"] = three({"wdhr": True}, "raw")
V["P2_wdhr_log"] = three({"wdhr": True})
for dg in (2, 3, 4, 5):
    V[f"P2_wdhr_deg{dg}_log"] = three({"wdhr": True, "deg": dg})
V["P2_wdhr_deg2_raw"] = three({"wdhr": True, "deg": 2}, "raw"); V["P2_wdhr_deg3_raw"] = three({"wdhr": True, "deg": 3}, "raw")
V["P2_wdhr_deg2_cross_log"] = three({"wdhr": True, "deg": 2, "cross": True})
V["P2_wdhr_deg3_th_log"] = three({"wdhr": True, "deg": 3, "pow_cols": ["temp", "hum"]})
V["P2_deg3_nowdhr_log"] = three({"deg": 3})
V["P2_wdhr_deg3_doy2_log"] = three({"wdhr": True, "deg": 3, "doy2": True})
out["variants"] = V

# smearing check on best log variant: with vs without
(_, _), smear, _, (Xt, y, w) = fit_eval(tr, [va], {"wdhr": True, "deg": 3})
Xv = build(tr, [va], {"wdhr": True, "deg": 3})[0][1]
out["smear_check"] = {"smear": smear, "val_r2_no_smear": float(r2(va.cnt.values, np.clip(np.expm1(Xv @ w), 0, None))),
                      "val_r2_smear": float(r2(va.cnt.values, np.clip(np.exp(Xv @ w) * smear - 1, 0, None))),
                      "val_mean_ratio_no": float(np.expm1(Xv @ w).mean() / va.cnt.mean()), "val_mean_ratio_sm": float((np.exp(Xv @ w) * smear - 1).mean() / va.cnt.mean())}


# ---------- D/E. conditioning and GD iteration counts ----------
def gd(X, y, w0, lr, tol=1e-10, max_iter=200000):
    n = len(y); w = w0.copy(); prev = None
    for it in range(1, max_iter + 1):
        r = X @ w - y; loss = (r @ r) / (2 * n); g = X.T @ r / n
        if prev is not None and abs(prev - loss) <= tol * max(prev, 1e-300) and np.linalg.norm(g) < 1e-6:
            return w, it, loss, "converged"
        prev = loss; w = w - lr * g
    return w, max_iter, loss, "max_iter"


C = {}
for name, opts in [("P1base", {}), ("P2_wdhr", {"wdhr": True}), ("P2_wdhr_deg2", {"wdhr": True, "deg": 2}), ("P2_wdhr_deg3", {"wdhr": True, "deg": 3}),
                   ("P2_wdhr_deg4", {"wdhr": True, "deg": 4})]:
    (Xs, names) = build(tr, [va], opts)
    Xt = Xs[0]; y = np.log1p(tr.cnt.values)
    ev = np.linalg.eigvalsh(Xt.T @ Xt / len(Xt))
    wo = np.linalg.lstsq(Xt, y, rcond=None)[0]
    lr = 1.0 / ev[-1]
    w, it, loss, why = gd(Xt, y, np.zeros(Xt.shape[1]), lr)
    C[name] = dict(p=len(names), lam_max=float(ev[-1]), lam_min=float(ev[0]), cond=float(ev[-1] / ev[0]), lr=float(lr), iters=it, stop=why,
                   max_abs_w_err=float(np.abs(w - wo).max()), loss_gap=float(loss - ((Xt @ wo - y) ** 2).sum() / (2 * len(y))))
out["conditioning"] = C

json.dump(out, open("exp/eda0/eda.json", "w"), indent=1, default=str)
print(json.dumps(out["facts"], default=str)[:3000])
print(json.dumps(out["split"]))
for k, v in V.items():
    print(f"{k:28s} {v}")
print(json.dumps(out["smear_check"]))
for k, v in C.items():
    print(k, v)
