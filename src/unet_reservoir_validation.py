"""
unet_reservoir_validation.py — E6: U-Net vs index/threshold water areas on the Chennai reservoirs,
validated against gauge storage, and their effect on satellite-driven forecasts.

Input : data/area_timeseries_unet.csv (stream_inference.py; 2017-2020, every scene, 6 methods
        on identical pixels: s2, s1, fused U-Nets; mndwi, ndwi, s1_otsu)
Output: outputs/unet_reservoir/

Rules
  * optical methods (s2, mndwi, ndwi) only on scenes with < CLOUD_MAX cloud over the reservoir,
    because they count cloud-free pixels only
  * "switch" = S2 U-Net on clear scenes, S1 U-Net on every other date (sensor switching)
  * rating curves are gauge-free for the target: leave-one-reservoir-out power law on
    capacity-normalised storage vs area / (that method's max area at that reservoir);
    CHOLAVARAM (unreliable gauge) is never used for fitting and is reported separately
  * forecasting: HistGB trained on 2004-2014 gauge data with satellite-like degraded inputs,
    applied with the satellite-derived state; scored against the true gauge

  python src/unet_reservoir_validation.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_experiments as fx
from area_to_storage import fit_power, power, time_blocked_cv
from metrics import regression_metrics
from satellite_forecast import test_frame

CLOUD_MAX = 0.10
FLAGGED = {"CHOLAVARAM"}
GOOD = [r for r in fx.RES if r not in FLAGGED]
OPTICAL = {"s2", "mndwi", "ndwi"}
LABEL = {"s2": "S2 U-Net", "s1": "S1 U-Net", "fused": "Fused U-Net", "mndwi": "MNDWI>0",
         "ndwi": "NDWI>0", "s1_otsu": "S1 Otsu", "switch": "Switch: S2 U-Net clear / S1 U-Net"}
SEASON = lambda m: np.where(m >= 10, "NE", np.where(m >= 6, "SW", "DRY"))


def load_areas(root, L, cfg):
    d = pd.read_csv(root / "data/area_timeseries_unet.csv", parse_dates=["date"])
    d = d[~(d.model.isin(OPTICAL) & (d.cloud_frac >= CLOUD_MAX))]
    d = d.groupby(["reservoir", "model", "date"], as_index=False).agg(area_km2=("area_km2", "mean"),
                                                                       area_std_km2=("area_std_km2", "mean"))
    d["storage"] = [L[r].get(t, np.nan) for r, t in zip(d.reservoir, d.date)]
    d["frac"] = d.storage / d.reservoir.map(lambda r: cfg[r]["capacity"])
    return d.dropna(subset=["storage"])


def loro_rating(d):
    """Gauge-free storage fraction for every row: power law fitted on OTHER, unflagged reservoirs."""
    d = d.copy(); d["frac_LORO"] = np.nan
    for m, g in d.groupby("model"):
        amax = g.groupby("reservoir").area_km2.max()
        for r in fx.RES:
            tr = g[(g.reservoir != r) & g.reservoir.isin(GOOD)]
            p = fit_power((tr.area_km2 / tr.reservoir.map(amax)).values, tr.frac.values)
            idx = g.index[g.reservoir == r]
            d.loc[idx, "frac_LORO"] = np.clip(power(d.loc[idx, "area_km2"].values / amax[r], *p), 0, 1.05)
    return d


def add_switch(d):
    s2 = d[d.model == "s2"].set_index(["reservoir", "date"])
    s1 = d[d.model == "s1"].set_index(["reservoir", "date"])
    sw = pd.concat([s2, s1[~s1.index.isin(s2.index)]]).reset_index().assign(model="switch")
    return pd.concat([d, sw], ignore_index=True)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/unet_reservoir"; (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "configs/reservoirs.json").read_text())
    L, R = fx.load(root / "data"); L = fx.qc_spikes(L, cfg)
    d = add_switch(loro_rating(load_areas(root, L, cfg)))
    d["season"] = SEASON(d.date.dt.month)
    d.to_csv(out / "areas_with_storage.csv", index=False)

    # ---------------------------------------------------------------- 1. area -> storage skill
    rows = []
    for (m, r), g in d.groupby(["model", "reservoir"]):
        g = g.sort_values("date"); cap = cfg[r]["capacity"]
        y, p = time_blocked_cv(g.area_km2.values, g.storage.values, g.date)
        ok = np.isfinite(p)
        lo = regression_metrics(g.frac, g.frac_LORO)
        rows.append({"method": LABEL[m], "reservoir": r, "n": len(g),
                     "spearman_rho": stats.spearmanr(g.area_km2, g.storage)[0],
                     "gauged_NSE": regression_metrics(y[ok] / cap, p[ok] / cap)["NSE"] if m != "switch" else np.nan,
                     "ungauged_NSE": lo["NSE"], "ungauged_RMSE_%cap": 100 * lo["RMSE"]})
    A = pd.DataFrame(rows); A.to_csv(out / "E6_area_vs_storage.csv", index=False)
    pooled = []
    for m, g in d[d.reservoir.isin(GOOD)].groupby("model"):
        for sn in ["ALL", "DRY", "SW", "NE"]:
            gg = g if sn == "ALL" else g[g.season == sn]
            met = regression_metrics(gg.frac, gg.frac_LORO)
            pooled.append({"method": LABEL[m], "season": sn, "n": len(gg),
                           "obs_per_reservoir_month": len(gg) / 3 / (48 if sn == "ALL" else {"DRY": 20, "SW": 16, "NE": 12}[sn]),
                           "NSE": met["NSE"], "RMSE_%cap": 100 * met["RMSE"]})
    PO = pd.DataFrame(pooled); PO.to_csv(out / "E6_pooled_by_season.csv", index=False)

    # ---------------------------------------------------------------- 2. dry-bed false water
    dry = []
    for (m, r), g in d[d.model != "switch"].groupby(["model", "reservoir"]):
        amax = g.area_km2.quantile(0.98)
        e = g[g.frac < 0.05]
        dry.append({"method": LABEL[m], "reservoir": r, "n_dry_scenes": len(e),
                    "dry_area_%_of_max": 100 * e.area_km2.median() / amax if len(e) else np.nan})
    DR = pd.DataFrame(dry); DR.to_csv(out / "E6_dry_bed_false_water.csv", index=False)

    # ---------------------------------------------------------------- 3. forecasting with each input
    chirps = pd.DataFrame({r: pd.read_csv(root / f"data/gee/rain_{r}.csv", parse_dates=["date"]).set_index("date").rain_mm
                           for r in fx.RES}).asfreq("D").fillna(0)
    D_gauge = fx.build(L, R, cfg)
    feats = [c for c in D_gauge.columns if c.startswith(("s_l", "d7", "d30", "r", "sin", "cos", "headroom"))
             and c not in ("res", "obs_age")]
    fs = feats + ["obs_age"]
    scen_opt = dict(revisit=10, cloud_loss=0.6, sar_fill=False, noise=0.08)
    scen_sar = dict(revisit=10, cloud_loss=0.6, sar_fill=True, noise=0.08)
    tr_g = D_gauge[D_gauge.split == "train"]
    tr_o = fx.build(L, R, cfg, scen_opt, seed=7); tr_o = tr_o[tr_o.split == "train"]
    tr_s = fx.build(L, R, cfg, scen_sar, seed=7); tr_s = tr_s[tr_s.split == "train"]
    S = d.rename(columns={"model": "sensor"})
    inputs = ["mndwi", "s2", "s1_otsu", "s1", "fused", "switch"]
    frows, preds = [], []
    windows = {"2017-2020": ("2017-04-01", "2020-11-03"), "2019-2020": ("2019-01-01", "2020-11-03")}
    import satellite_forecast as sf
    for wname, (w0, w1) in windows.items():
        sf.TEST_START, sf.TEST_END = w0, w1
        for h in fx.HORIZONS:
            fit = lambda tr, f: fx.models()["HistGB"]().fit(tr.dropna(subset=[f"y{h}"])[f], tr.dropna(subset=[f"y{h}"])[f"y{h}"])
            mG, mO, mS = fit(tr_g, feats), fit(tr_o, fs), fit(tr_s, fs)
            for r in fx.RES:
                Dg = D_gauge[D_gauge.res == r].loc[w0:w1].dropna(subset=[f"y{h}"])
                truth = pd.Series(fx.truth(Dg, h), Dg.index)
                res = {"Gauge input (reference)": pd.Series(fx.to_storage(Dg, mG.predict(Dg[feats]), h), Dg.index)}
                for m in inputs:
                    T = test_frame(L, R, S, cfg, r, m).dropna(subset=[f"y{h}"])
                    mdl = mO if m in OPTICAL else mS
                    res[LABEL[m]] = pd.Series(fx.to_storage(T, mdl.predict(T[fs]), h), T.index)
                    if m == "switch":
                        res["Persistence of switch"] = pd.Series(T.s_l0.values * T.cap.values, T.index)
                # fully gauge-free: forecaster trained on the other reservoirs, CHIRPS rain, switch input
                mC = fit(tr_s[tr_s.res != r], fs)
                Tc = test_frame(L, chirps, S, cfg, r, "switch").dropna(subset=[f"y{h}"])
                res["Gauge-free: switch + CHIRPS, unseen reservoir"] = pd.Series(fx.to_storage(Tc, mC.predict(Tc[fs]), h), Tc.index)
                common = truth.index
                for s_ in res.values():
                    common = common.intersection(s_.index)
                preds.append(pd.DataFrame({"window": wname, "h": h, "reservoir": r, "date": common,
                                           "truth": truth[common].values, **{k: v[common].values for k, v in res.items()}}))
    P = pd.concat(preds); P.to_csv(out / "E6_forecast_predictions.csv", index=False)
    long = P.melt(id_vars=["window", "h", "reservoir", "date", "truth"], var_name="input", value_name="pred")
    long["cap"] = long.reservoir.map(lambda r: cfg[r]["capacity"])
    for (w, h, inp), g in long[long.reservoir.isin(GOOD)].groupby(["window", "h", "input"]):
        met = regression_metrics(g.truth / g.cap, g.pred / g.cap)
        frows.append({"window": w, "horizon": h, "input": inp, "NSE": met["NSE"], "RMSE_%cap": 100 * met["RMSE"]})
    F = pd.DataFrame(frows); F.to_csv(out / "E6_forecast_metrics_pooled.csv", index=False)

    figures(out, d, cfg, L, DR, P)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(A.pivot_table(index="method", columns="reservoir", values="ungauged_NSE").round(3))
    print(A.pivot_table(index="method", columns="reservoir", values="spearman_rho").round(3))
    print(PO.pivot_table(index="method", columns="season", values="NSE").round(3))
    print(PO.pivot_table(index="method", columns="season", values="obs_per_reservoir_month").round(2))
    print(DR.pivot_table(index="method", columns="reservoir", values="dry_area_%_of_max").round(1))
    for w in windows:
        print("forecast NSE", w); print(F[F.window == w].pivot(index="input", columns="horizon", values="NSE").round(3))


def figures(out, d, cfg, L, DR, P):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    col = {"s1_otsu": "#e3a72f", "s1": "#c0392b", "mndwi": "#7fb3d5", "s2": "#1f5aa6", "fused": "#2e9e6b"}
    # area vs storage scatter, SAR methods
    f, ax = plt.subplots(2, 4, figsize=(11, 5))
    for j, r in enumerate(fx.RES):
        for i, (ms, ttl) in enumerate([(("s1_otsu", "s1"), "SAR"), (("mndwi", "s2"), "Optical (clear)")]):
            for m in ms:
                g = d[(d.model == m) & (d.reservoir == r)]
                ax[i, j].scatter(g.area_km2, g.frac * 100, s=6, alpha=.55, color=col[m],
                                 label=f"{LABEL[m]} (ρ={stats.spearmanr(g.area_km2, g.storage)[0]:.2f})")
            ax[i, j].set_title(f"{r.title()} — {ttl}", fontsize=8); ax[i, j].legend(frameon=False, fontsize=6)
            ax[i, j].set_xlabel("Water area (km²)"); ax[i, j].set_ylabel("Gauge storage (% cap)")
    plt.tight_layout(); plt.savefig(out / "figures/U1_area_vs_storage_unet_vs_baselines.png"); plt.close()
    # time series: switch-derived storage vs gauge
    f, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for i, r in enumerate(fx.RES):
        cap = cfg[r]["capacity"]
        ax[i].plot(L.loc["2017":"2020"].index, L.loc["2017":"2020", r] / cap * 100, "k", lw=1, label="Gauge")
        for m, mk in [("s1_otsu", "x"), ("s1", "o"), ("s2", "^")]:
            g = d[(d.model == m) & (d.reservoir == r)]
            ax[i].plot(g.date, g.frac_LORO * 100, mk, ms=2.5, color=col[m], label=LABEL[m], alpha=.85)
        ax[i].set_ylabel(f"{r.title()}\n% capacity" + (" (gauge flagged)" if r in FLAGGED else "")); ax[i].set_ylim(-5, 110)
    ax[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    ax[0].set_title("Satellite-derived storage, gauge-free rating curves (fitted on other reservoirs)")
    plt.savefig(out / "figures/U2_storage_timeseries_unet.png"); plt.close()
    # forecasts
    h = 15; g0 = P[(P.window == "2017-2020") & (P.h == h)]
    f, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for i, r in enumerate(GOOD):
        g = g0[g0.reservoir == r].sort_values("date"); cap = cfg[r]["capacity"]
        t = pd.to_datetime(g.date) + pd.Timedelta(days=h)
        ax[i].plot(t, g.truth / cap * 100, "k", lw=1.1, label="Gauge (truth)")
        ax[i].plot(t, g["MNDWI>0"] / cap * 100, color="#7fb3d5", lw=.8, label="MNDWI input")
        ax[i].plot(t, g[LABEL["switch"]] / cap * 100, color="#c0392b", lw=.9, label="U-Net switch input")
        ax[i].set_ylabel(f"{r.title()}\n% capacity")
    ax[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax[0].set_title(f"{h}-day-ahead forecasts driven by satellite storage")
    plt.savefig(out / "figures/U3_forecasts_unet.png"); plt.close()


if __name__ == "__main__":
    main()
