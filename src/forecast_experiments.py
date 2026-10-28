"""
forecast_experiments.py — reservoir storage forecasting with calibrated uncertainty.

Runs on the public Kaggle "Chennai Water Management" data (daily storage in mcft and rainfall
in mm, 2004-2020, for POONDI, CHOLAVARAM, REDHILLS, CHEMBARAMBAKKAM).

Experiments (all chronological; no test data is touched before final scoring):
  E1  Point forecasts, horizons 7/15/30 d: persistence, climatology, AR-ridge, RF, HistGB, MLP
      Metrics: RMSE, MAE (mcft), NSE, KGE, PBIAS; Diebold-Mariano test vs persistence.
  E2  Uncertainty at 90 %: split conformal, Mondrian conformal (season x reservoir),
      adaptive conformal (ACI, with the h-day delayed feedback a real forecaster has),
      vs. quantile-HistGB. Metrics: PICP, MPIW, Winkler.
  E3  Transfer to an unseen reservoir (leave-one-reservoir-out, LORO): model trained only on
      the other three, applied to the fourth using capacity-normalised storage.
  E4  Satellite-observation sensitivity (SIMULATION): the input state is degraded to mimic
      satellite-derived storage (revisit interval, monsoon cloud loss, SAR gap-filling,
      area->storage error). Targets stay the true gauge values.

Splits: train 2004-2014 | calibration 2015-2016 | test 2017-2020 (the Sentinel era).

  python src/forecast_experiments.py --data data --config configs/reservoirs.json --out outputs/forecast
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import interval_metrics, regression_metrics

warnings.filterwarnings("ignore")
RES = ["POONDI", "CHOLAVARAM", "REDHILLS", "CHEMBARAMBAKKAM"]
HORIZONS = [7, 15, 30]
ALPHA = 0.10
TRAIN_END, CAL_END = "2014-12-31", "2016-12-31"
SEED = 0


def season(month):
    # Tamil Nadu: dry Jan-May, SW monsoon Jun-Sep, NE monsoon Oct-Dec (main inflow season)
    return np.where(month >= 10, "NE", np.where(month >= 6, "SW", "DRY"))


# ----------------------------------------------------------------------------- data
def load(data_dir):
    L = pd.read_csv(Path(data_dir) / "chennai_reservoir_levels.csv")
    R = pd.read_csv(Path(data_dir) / "chennai_reservoir_rainfall.csv")
    for d in (L, R):
        d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
    L = L.set_index("Date").asfreq("D"); R = R.set_index("Date").asfreq("D")
    return L.interpolate(limit=3), R.fillna(0.0)


def qc_spikes(L, cfg, out=None, jump=0.15, agree=0.05, max_len=5):
    """Flag isolated blocks of 1..max_len days (data-entry errors) that jump by > `jump` x capacity
    away from the series and back, while the values just before and just after the block agree
    within `agree` x capacity. Flagged values are replaced by linear interpolation."""
    L = L.copy(); log = []
    for r in RES:
        cap = cfg[r]["capacity"]; x = L[r].values.astype(float).copy(); n = len(x)
        bad = np.zeros(n, bool)
        for w in range(1, max_len + 1):
            for t in range(1, n - w):
                a, b = x[t - 1], x[t + w]
                blk = x[t:t + w]
                if abs(a - b) < agree * cap and np.all(np.abs(blk - a) > jump * cap) \
                        and np.all(np.abs(blk - b) > jump * cap) and not bad[t - 1]:
                    bad[t:t + w] = True
        for t in np.where(bad)[0]:
            log.append({"reservoir": r, "date": L.index[t].date(), "value": x[t]})
        L[r] = pd.Series(np.where(bad, np.nan, x), index=L.index).interpolate()
    if out is not None:
        pd.DataFrame(log).to_csv(Path(out) / "qc_flagged_spikes.csv", index=False)
    print(f"QC: replaced {len(log)} spike values", flush=True)
    return L


def observe(s, scenario, rng):
    """Return the storage series a forecaster would SEE (s is the true daily fraction).
    scenario: dict(revisit, cloud_loss, sar_fill, noise)  — None = daily gauge."""
    if scenario is None:
        return s.copy(), np.zeros(len(s))
    n = len(s); idx = np.arange(n)
    month = s.index.month.values
    obs = np.zeros(n, bool)
    opt = (idx % scenario["revisit"]) == 0
    wet = (month >= 6)  # Jun-Dec: heavy cloud
    opt &= ~(wet & (rng.random(n) < scenario["cloud_loss"]))
    obs |= opt
    if scenario["sar_fill"]:
        obs |= (idx % 12) == 3  # Sentinel-1 12-day repeat, cloud-independent
    vals = s.values + rng.normal(0, scenario["noise"], n)
    seen = pd.Series(np.where(obs, vals, np.nan), index=s.index).ffill().bfill()
    last = pd.Series(np.where(obs, idx, np.nan)).ffill().fillna(0).values
    return seen.clip(0, 1.1), idx - last  # value, days since last observation


def features(st, rain_own, rain_all, age):
    f = pd.DataFrame(index=st.index)
    for l in [0, 1, 3, 7, 14, 30, 60]:
        f[f"s_l{l}"] = st.shift(l)
    f["d7"] = st - st.shift(7); f["d30"] = st - st.shift(30)
    for w in [1, 3, 7, 15, 30, 60, 90]:
        f[f"r{w}"] = rain_own.rolling(w).sum()
    for w in [7, 30]:
        f[f"ra{w}"] = rain_all.rolling(w).sum()
    doy = st.index.dayofyear.values
    f["sin"] = np.sin(2 * np.pi * doy / 365.25); f["cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["headroom"] = 1 - st
    f["obs_age"] = age
    return f


def build(L, R, cfg, scenario=None, seed=SEED):
    rng = np.random.default_rng(seed)
    rain_all = R[RES].mean(1)
    out = []
    for r in RES:
        cap = cfg[r]["capacity"]
        s_true = (L[r] / cap).clip(0, 1.1)
        s_seen, age = observe(s_true, scenario, rng)
        F = features(s_seen, R[r], rain_all, age)
        F["res"] = r; F["cap"] = cap; F["s_true"] = s_true
        for h in HORIZONS:
            F[f"y{h}"] = s_true.shift(-h) - s_seen  # change relative to what is seen now
        out.append(F)
    D = pd.concat(out).dropna(subset=["s_l60", "r90"])
    D["season"] = season(D.index.month.values)
    D["split"] = np.where(D.index <= TRAIN_END, "train", np.where(D.index <= CAL_END, "cal", "test"))
    return D


FEATS = None  # set after build


def models():
    return {
        "AR-Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                      n_jobs=-1, random_state=SEED),
        "HistGB": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                        max_leaf_nodes=31, l2_regularization=1.0,
                                                        random_state=SEED),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                     learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                                     random_state=SEED)),
    }


def to_storage(D, yhat, h):
    """change-in-fraction -> storage (mcft), clipped to physical range."""
    s_seen = D["s_l0"].values
    return np.clip(s_seen + yhat, 0, 1.1) * D["cap"].values


def truth(D, h):
    return (D["s_l0"].values + D[f"y{h}"].values) * D["cap"].values


def climatology_change(Dtr, D, h):
    clim = Dtr.groupby([Dtr["res"], Dtr.index.dayofyear])[f"y{h}"].mean()
    key = list(zip(D["res"], D.index.dayofyear))
    return np.array([clim.get(k, 0.0) for k in key])


def dm_test(e1, e2, h):
    """Diebold-Mariano (HLN small-sample corrected), squared loss; e1 = benchmark errors.
    Positive stat => model 2 better."""
    d = e1 ** 2 - e2 ** 2
    n = len(d); dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), n - 1)))


# ----------------------------------------------------------------------------- conformal
def conformal_q(res_abs, alpha):
    n = len(res_abs)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(res_abs)[k - 1]


def aci(test_pred, test_true, cal_abs, h, alpha=ALPHA, gamma=0.005):
    """Adaptive conformal inference (Gibbs & Candes 2021) with delayed feedback:
    the error of the forecast issued at t is only known at t+h."""
    n = len(test_pred); a_t = alpha; lo = np.empty(n); hi = np.empty(n)
    scores = list(np.sort(cal_abs)); errs = np.zeros(n, bool); alphas = np.empty(n)
    for t in range(n):
        if t - h >= 0:  # feedback for forecast issued h steps ago
            a_t = a_t + gamma * (alpha - errs[t - h])
            scores.append(abs(test_true[t - h] - test_pred[t - h]))
        a_c = float(np.clip(a_t, 1e-3, 0.999))
        q = np.quantile(scores, 1 - a_c)
        lo[t], hi[t] = test_pred[t] - q, test_pred[t] + q
        errs[t] = not (lo[t] <= test_true[t] <= hi[t]); alphas[t] = a_c
    return lo, hi


# ----------------------------------------------------------------------------- experiments
def run(a):
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    L, R = load(a.data)
    L = qc_spikes(L, cfg, out)
    D = build(L, R, cfg)
    global FEATS
    FEATS = [c for c in D.columns if c.startswith(("s_l", "d7", "d30", "r", "sin", "cos", "headroom"))
             and c not in ("res",)]
    FEATS = [c for c in FEATS if c != "obs_age"]
    tr, ca, te = (D[D.split == s] for s in ("train", "cal", "test"))
    print({k: len(v) for k, v in {"train": tr, "cal": ca, "test": te}.items()}, "features", len(FEATS))

    point_rows, dm_rows, int_rows, pred_store = [], [], [], {}
    t0 = time.time()
    for h in HORIZONS:
        mtr = tr.dropna(subset=[f"y{h}"]); mca = ca.dropna(subset=[f"y{h}"]); mte = te.dropna(subset=[f"y{h}"])
        y_true = truth(mte, h)
        preds = {"Persistence": to_storage(mte, np.zeros(len(mte)), h),
                 "Climatology": to_storage(mte, climatology_change(mtr, mte, h), h)}
        cal_preds = {}
        for name, mk in models().items():
            m = mk().fit(mtr[FEATS], mtr[f"y{h}"])
            preds[name] = to_storage(mte, m.predict(mte[FEATS]), h)
            cal_preds[name] = to_storage(mca, m.predict(mca[FEATS]), h)
            print(f"  h={h} {name} done ({time.time()-t0:.0f}s)", flush=True)
        # point metrics per reservoir + pooled (pooled in capacity fraction for NSE comparability)
        for name, p in preds.items():
            for r in RES + ["ALL"]:
                msk = (mte.res == r).values if r != "ALL" else np.ones(len(mte), bool)
                met = regression_metrics(y_true[msk], p[msk])
                if r == "ALL":  # pooled metrics on capacity-normalised storage
                    met = regression_metrics(y_true[msk] / mte.cap.values[msk], p[msk] / mte.cap.values[msk])
                point_rows.append({"horizon": h, "model": name, "reservoir": r, **met})
            for sn in ["DRY", "SW", "NE"]:  # pooled skill by season of the forecast issue date
                msk = (mte.season == sn).values
                met = regression_metrics(y_true[msk] / mte.cap.values[msk], p[msk] / mte.cap.values[msk])
                point_rows.append({"horizon": h, "model": name, "reservoir": f"ALL_{sn}", **met})
            if name != "Persistence":
                for r in RES:
                    msk = (mte.res == r).values
                    stat, pv = dm_test(y_true[msk] - preds["Persistence"][msk], y_true[msk] - p[msk], h)
                    dm_rows.append({"horizon": h, "model": name, "reservoir": r, "DM_stat": stat, "p_value": pv})
        pred_store[h] = {"mte": mte, "y": y_true, "preds": preds}
        pd.DataFrame({"issue_date": mte.index, "reservoir": mte.res.values, "obs_mcft": y_true,
                      **{f"pred_{k}": v for k, v in preds.items()}}).to_csv(out / f"predictions_h{h}.csv", index=False)

        # ---------------- E2 intervals for the two best tree models + MLP ----------------
        ycal = truth(mca, h)
        for name in ["HistGB", "RandomForest", "MLP"]:
            p_te, p_ca = preds[name], cal_preds[name]
            # (a) split conformal, pooled — on capacity-fraction scale, rescaled per reservoir
            sc = np.abs(ycal - p_ca) / mca.cap.values
            q = conformal_q(sc, ALPHA)
            lo, hi = p_te - q * mte.cap.values, p_te + q * mte.cap.values
            store_int(int_rows, h, name, "Split-CP", mte, y_true, lo, hi)
            # (b) Mondrian: season x reservoir
            lo, hi = np.empty(len(mte)), np.empty(len(mte))
            for (r, sn), _ in mte.groupby(["res", "season"]):
                cm = ((mca.res == r) & (mca.season == sn)).values
                tm = ((mte.res == r) & (mte.season == sn)).values
                qq = conformal_q(np.abs(ycal[cm] - p_ca[cm]), ALPHA)
                lo[tm], hi[tm] = p_te[tm] - qq, p_te[tm] + qq
            store_int(int_rows, h, name, "Mondrian-CP", mte, y_true, lo, hi)
            if name == "HistGB":
                pred_store[h]["mondrian"] = (lo.copy(), hi.copy())
            # (c) ACI per reservoir with delayed feedback (Mondrian scores as warm start)
            lo, hi = np.empty(len(mte)), np.empty(len(mte))
            for r in RES:
                tm = (mte.res == r).values; cm = (mca.res == r).values
                lo[tm], hi[tm] = aci(p_te[tm], y_true[tm], np.abs(ycal[cm] - p_ca[cm]), h)
            store_int(int_rows, h, name, "ACI", mte, y_true, lo, hi)
            if name == "HistGB":
                pred_store[h]["aci"] = (lo.copy(), hi.copy())
        # (d) quantile HistGB baseline
        qlo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, max_iter=400,
                                            learning_rate=0.05, random_state=SEED).fit(mtr[FEATS], mtr[f"y{h}"])
        qhi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, max_iter=400,
                                            learning_rate=0.05, random_state=SEED).fit(mtr[FEATS], mtr[f"y{h}"])
        store_int(int_rows, h, "HistGB", "Quantile-reg", mte, y_true,
                  to_storage(mte, qlo.predict(mte[FEATS]), h), to_storage(mte, qhi.predict(mte[FEATS]), h))

    P = pd.DataFrame(point_rows); P.to_csv(out / "E1_point_metrics.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(out / "E1_dm_tests.csv", index=False)
    I = pd.DataFrame(int_rows); I.to_csv(out / "E2_interval_metrics.csv", index=False)

    # ---------------- E3 leave-one-reservoir-out ----------------
    loro = []
    for h in HORIZONS:
        mtr = tr.dropna(subset=[f"y{h}"]); mte = te.dropna(subset=[f"y{h}"])
        for r in RES:
            te_r = mte[mte.res == r]; yt = truth(te_r, h)
            for setting, train_set in [("Local (own data)", mtr[mtr.res == r]),
                                       ("Pooled (all 4)", mtr),
                                       ("LORO (unseen)", mtr[mtr.res != r])]:
                m = models()["HistGB"]().fit(train_set[FEATS], train_set[f"y{h}"])
                loro.append({"horizon": h, "reservoir": r, "setting": setting,
                             **regression_metrics(yt, to_storage(te_r, m.predict(te_r[FEATS]), h))})
            loro.append({"horizon": h, "reservoir": r, "setting": "Persistence",
                         **regression_metrics(yt, to_storage(te_r, np.zeros(len(te_r)), h))})
    LO = pd.DataFrame(loro); LO.to_csv(out / "E3_loro.csv", index=False)
    print(f"E3 done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- E4 satellite-observation sensitivity (simulation) ----------------
    scen = {
        "Daily gauge": None,
        "Optical only (5 d, cloudy monsoon)": dict(revisit=5, cloud_loss=0.7, sar_fill=False, noise=0.03),
        "SAR+optical (5 d + S1 12 d)":        dict(revisit=5, cloud_loss=0.7, sar_fill=True, noise=0.03),
    }
    sens = []
    for noise in [0.0, 0.02, 0.05, 0.10]:
        for sname, sc in scen.items():
            if sc is None and noise > 0:
                continue
            scc = None if sc is None else {**sc, "noise": noise}
            for rep in range(3 if scc else 1):
                Ds = build(L, R, cfg, scc, seed=100 + rep)
                trs, tes = Ds[Ds.split == "train"], Ds[Ds.split == "test"]
                feats = FEATS + (["obs_age"] if scc else [])
                for h in HORIZONS:
                    a_, b_ = trs.dropna(subset=[f"y{h}"]), tes.dropna(subset=[f"y{h}"])
                    m = models()["HistGB"]().fit(a_[feats], a_[f"y{h}"])
                    p = to_storage(b_, m.predict(b_[feats]), h); yt = b_["s_true"].shift(0)  # placeholder
                    # truth = true storage at t+h = (seen + y) * cap
                    yt = truth(b_, h)
                    met = regression_metrics(yt / b_.cap.values, p / b_.cap.values)
                    sens.append({"scenario": sname, "noise_frac": noise, "rep": rep, "horizon": h, **met})
    S = pd.DataFrame(sens); S.to_csv(out / "E4_satellite_sensitivity.csv", index=False)
    print(f"E4 done ({time.time()-t0:.0f}s)", flush=True)

    figures(out, L, R, cfg, P, I, LO, S, pred_store)
    summary(out, P, pd.DataFrame(dm_rows), I, LO, S)


def store_int(rows, h, model, method, mte, y, lo, hi):
    lo, hi = phys_clip(mte, lo, hi)
    for r in RES + ["ALL"]:
        msk = (mte.res == r).values if r != "ALL" else np.ones(len(mte), bool)
        im = interval_metrics(y[msk], lo[msk], hi[msk], ALPHA)
        if r == "ALL":
            im["MPIW"] = float(((hi - lo)[msk] / mte.cap.values[msk]).mean())  # fraction of capacity
        ne = msk & (mte.season == "NE").values
        rows.append({"horizon": h, "model": model, "method": method, "reservoir": r, **im,
                     "PICP_NEmonsoon": float(((y >= lo) & (y <= hi))[ne].mean())})


def phys_clip(mte, lo, hi):
    """Storage cannot be negative or exceed ~full reservoir level (+5 % surcharge)."""
    cap = mte.cap.values
    return np.clip(lo, 0, 1.05 * cap), np.clip(hi, 0, 1.05 * cap)


# ----------------------------------------------------------------------------- figures
def figures(out, L, R, cfg, P, I, LO, S, store):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    col = {"Persistence": "#888888", "Climatology": "#bbbbbb", "AR-Ridge": "#e3a72f",
           "RandomForest": "#3b7dd8", "HistGB": "#c0392b", "MLP": "#2e9e6b"}
    fig = out / "figures"

    # F1 data overview
    f, ax = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    for i, r in enumerate(RES):
        ax[i].plot(L.index, L[r] / cfg[r]["capacity"] * 100, color="#1f5aa6", lw=.7)
        ax2 = ax[i].twinx(); ax2.bar(R.index, R[r], color="#7fb3d5", width=1, alpha=.6)
        ax2.set_ylim(0, 250); ax2.set_yticks([]); ax2.spines["top"].set_visible(False)
        ax[i].set_ylabel(f"{r.title()}\n% capacity"); ax[i].set_ylim(0, 110)
        for x, c in [(TRAIN_END, "k"), (CAL_END, "k")]:
            ax[i].axvline(pd.Timestamp(x), color=c, ls="--", lw=.7)
    ax[0].set_title("Storage (line) and daily rainfall (bars); dashed = train | calibration | test")
    plt.savefig(fig / "F1_data_overview.png"); plt.close()

    # F2 skill vs horizon
    f, ax = plt.subplots(1, 2, figsize=(8, 3))
    A = P[P.reservoir == "ALL"]
    for m, g in A.groupby("model"):
        ax[0].plot(g.horizon, g.NSE, "o-", color=col[m], label=m, lw=1.5)
        ax[1].plot(g.horizon, g.RMSE * 100, "o-", color=col[m], lw=1.5)
    ax[0].set(xlabel="Lead time (days)", ylabel="NSE (pooled)", xticks=HORIZONS)
    ax[1].set(xlabel="Lead time (days)", ylabel="RMSE (% of capacity)", xticks=HORIZONS)
    ax[0].legend(frameon=False, fontsize=7); plt.tight_layout()
    plt.savefig(fig / "F2_skill_vs_horizon.png"); plt.close()

    # F3 30-day forecast with intervals, NE monsoon 2017-2020, Chembarambakkam
    h = 30; st = store[h]; mte = st["mte"]; r = "CHEMBARAMBAKKAM"
    msk = (mte.res == r).values
    t = mte.index[msk] + pd.Timedelta(days=h)
    lo, hi = phys_clip(mte, *st["mondrian"])
    f, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between(t, lo[msk], hi[msk], color="#c0392b", alpha=.18, lw=0, label="90 % Mondrian conformal interval")
    ax.plot(t, st["y"][msk], color="k", lw=.9, label="Observed")
    ax.plot(t, st["preds"]["HistGB"][msk], color="#c0392b", lw=.9, label="HistGB 30-d forecast")
    ax.plot(t, st["preds"]["Persistence"][msk], color="#888", lw=.7, ls="--", label="Persistence")
    ax.set(ylabel="Storage (mcft)", title=f"{r.title()} — 30-day-ahead forecasts on the test period")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left"); plt.savefig(fig / "F3_forecast_intervals.png"); plt.close()

    # F4 coverage: overall vs NE monsoon by method (HistGB)
    f, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    J = I[(I.model == "HistGB") & (I.reservoir == "ALL")]
    meths = ["Quantile-reg", "Split-CP", "Mondrian-CP", "ACI"]
    for i, h in enumerate(HORIZONS):
        g = J[J.horizon == h].set_index("method").loc[meths]
        x = np.arange(len(meths))
        ax[i].bar(x - .18, g.PICP * 100, .36, color="#3b7dd8", label="All test days")
        ax[i].bar(x + .18, g.PICP_NEmonsoon * 100, .36, color="#e3a72f", label="NE monsoon only")
        ax[i].axhline(90, color="k", lw=.8, ls="--"); ax[i].set_xticks(x, meths, rotation=30, ha="right")
        ax[i].set_title(f"{h}-day lead"); ax[i].set_ylim(40, 100)
    ax[0].set_ylabel("Interval coverage (%)"); ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    plt.savefig(fig / "F4_interval_coverage.png"); plt.close()

    # F5 LORO
    f, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    sets = ["Persistence", "Local (own data)", "Pooled (all 4)", "LORO (unseen)"]
    cc = ["#888", "#3b7dd8", "#2e9e6b", "#c0392b"]
    for i, h in enumerate(HORIZONS):
        g = LO[LO.horizon == h]
        x = np.arange(len(RES))
        for j, s in enumerate(sets):
            v = g[g.setting == s].set_index("reservoir").loc[RES].NSE
            ax[i].bar(x + (j - 1.5) * .2, v, .2, color=cc[j], label=s)
        ax[i].set_xticks(x, [r.title()[:9] for r in RES], rotation=30, ha="right"); ax[i].set_title(f"{h}-day lead")
    ax[0].set_ylabel("NSE"); ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    plt.savefig(fig / "F5_transfer_loro.png"); plt.close()

    # F6 satellite sensitivity
    f, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    G = S.groupby(["scenario", "noise_frac", "horizon"]).NSE.agg(["mean", "std"]).reset_index()
    sc_col = {"Optical only (5 d, cloudy monsoon)": "#e3a72f", "SAR+optical (5 d + S1 12 d)": "#c0392b"}
    for i, h in enumerate(HORIZONS):
        base = G[(G.scenario == "Daily gauge") & (G.horizon == h)]["mean"].iloc[0]
        ax[i].axhline(base, color="k", ls="--", lw=.8, label="Daily gauge")
        for s, c in sc_col.items():
            g = G[(G.scenario == s) & (G.horizon == h)]
            ax[i].errorbar(g.noise_frac * 100, g["mean"], g["std"], color=c, marker="o", lw=1.4, capsize=2, label=s)
        ax[i].set(title=f"{h}-day lead", xlabel="Satellite storage error (% capacity, 1σ)")
    ax[0].set_ylabel("NSE (pooled)"); ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    plt.savefig(fig / "F6_satellite_sensitivity.png"); plt.close()


def summary(out, P, DM, I, LO, S):
    f = lambda d: d.to_markdown(index=False, floatfmt=".3f") if hasattr(d, "to_markdown") else d.to_string(index=False)
    lines = ["# Results summary (auto-generated)\n"]
    A = P[P.reservoir == "ALL"].pivot(index="model", columns="horizon", values="NSE").round(3)
    lines += ["## E1 pooled NSE (capacity-normalised)\n", A.to_string(), ""]
    A2 = P[P.reservoir.str.startswith("ALL_")].pivot_table(index=["reservoir", "model"], columns="horizon", values="NSE").round(3)
    lines += ["## E1 pooled NSE by season\n", A2.to_string(), ""]
    A3 = P[P.reservoir.str.startswith("ALL_")].pivot_table(index=["reservoir", "model"], columns="horizon", values="RMSE").round(4)
    lines += ["## E1 pooled RMSE by season (fraction of capacity)\n", A3.to_string(), ""]
    B = P[P.reservoir != "ALL"].pivot_table(index=["reservoir", "model"], columns="horizon", values="RMSE").round(1)
    lines += ["## E1 RMSE (mcft) per reservoir\n", B.to_string(), ""]
    lines += ["## E1 Diebold-Mariano vs persistence (HistGB)\n",
              DM[DM.model == "HistGB"].round(4).to_string(index=False), ""]
    C = I[I.reservoir == "ALL"][["horizon", "model", "method", "PICP", "PICP_NEmonsoon", "MPIW", "Winkler"]].round(3)
    lines += ["## E2 90 % intervals (ALL; MPIW = fraction of capacity)\n", C.to_string(index=False), ""]
    D = LO.pivot_table(index=["reservoir", "setting"], columns="horizon", values="NSE").round(3)
    lines += ["## E3 transfer NSE\n", D.to_string(), ""]
    E = S.groupby(["scenario", "noise_frac", "horizon"]).NSE.agg(["mean", "std"]).round(3).reset_index()
    lines += ["## E4 satellite sensitivity, pooled NSE (SIMULATED observations)\n", E.to_string(index=False), ""]
    (out / "results_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--config", default="configs/reservoirs.json")
    ap.add_argument("--out", default="outputs/forecast")
    run(ap.parse_args())
