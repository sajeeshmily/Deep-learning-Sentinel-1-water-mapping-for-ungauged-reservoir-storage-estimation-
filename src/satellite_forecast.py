"""
satellite_forecast.py — E5: storage forecasting driven by REAL satellite observations.

The forecaster is trained on the 2004-2014 gauge record, with its input state degraded to
mimic satellite sampling (revisit, monsoon cloud loss, storage error, "days since last
observation" feature). It is then run on 2019-2020 with the REAL satellite-derived storage
as its only view of the reservoir; forecasts are scored against the true gauge.

Satellite storage = S2 (or S1) area through the leave-one-reservoir-out (LORO) rating curve,
i.e. no gauge data from the target reservoir is used to convert area to storage.

Configurations per target reservoir:
  A  gauge input, forecaster trained on all 4               (reference, needs a gauge)
  B  S2 input,   forecaster trained on all 4 (gauge rain)   (satellite state, gauged elsewhere)
  C  S2 input,   forecaster trained on the other 3 + CHIRPS rainfall  -> FULLY GAUGE-FREE
  D  S1 (Otsu) input, as B                                  (shows thresholded SAR fails)
  P  persistence of the last satellite storage value        (gauge-free baseline)
Test window 2019-01-01 .. 2020-11-03 (S2 L2A archive complete; truth needed 30 d ahead).

  python src/satellite_forecast.py   (run satellite_validation.py first)
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_experiments as fx
from metrics import regression_metrics

TEST_START, TEST_END = "2019-01-01", "2020-11-03"
TRAIN_SCEN = dict(revisit=10, cloud_loss=0.6, sar_fill=False, noise=0.08)  # matches real S2 availability / LORO error


def sat_daily(S, r, sensor, index):
    g = S[(S.reservoir == r) & (S.sensor == sensor)].sort_values("date")
    obs = pd.Series(g.frac_LORO.values, index=g.date).groupby(level=0).mean().reindex(index)
    seen = obs.ffill()
    pos = np.arange(len(index))
    last = pd.Series(np.where(obs.notna(), pos, np.nan)).ffill().values
    return seen, pd.Series(pos - last, index=index)


def test_frame(L, R_rain, S, cfg, r, sensor):
    idx = pd.date_range("2017-01-01", L.index.max(), freq="D")
    seen, age = sat_daily(S, r, sensor, idx)
    rain_all = pd.concat([R_rain[x] for x in fx.RES], axis=1).mean(1).reindex(idx).fillna(0)
    F = fx.features(seen, R_rain[r].reindex(idx).fillna(0), rain_all, age)
    s_true = (L[r] / cfg[r]["capacity"]).reindex(idx)
    F["res"] = r; F["cap"] = cfg[r]["capacity"]; F["s_true"] = s_true
    for h in fx.HORIZONS:
        F[f"y{h}"] = s_true.shift(-h) - seen
    return F.loc[TEST_START:TEST_END].dropna(subset=["s_l60", "r90"])


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/satellite"
    cfg = json.loads((root / "configs/reservoirs.json").read_text())
    L, R = fx.load(root / "data"); L = fx.qc_spikes(L, cfg)
    S = pd.read_csv(out / "satellite_areas_clean.csv", parse_dates=["date"])

    # CHIRPS catchment rainfall (gauge-free forcing), 2017-2020
    chirps = pd.DataFrame({r: pd.read_csv(root / f"data/gee/rain_{r}.csv", parse_dates=["date"])
                           .set_index("date").rain_mm for r in fx.RES}).asfreq("D").fillna(0)
    m_g = R.loc["2017":"2020"].resample("ME").sum(); m_c = chirps.resample("ME").sum().reindex(m_g.index)
    chirps_corr = {r: round(float(np.corrcoef(m_g[r], m_c[r])[0, 1]), 3) for r in fx.RES}
    print("monthly rain corr gauge vs CHIRPS:", chirps_corr)

    # training sets (2004-2014 gauge record)
    D_gauge = fx.build(L, R, cfg)
    feats = [c for c in D_gauge.columns if c.startswith(("s_l", "d7", "d30", "r", "sin", "cos", "headroom"))
             and c not in ("res", "obs_age")]
    D_sat = fx.build(L, R, cfg, TRAIN_SCEN, seed=7)
    tr_g, tr_s = D_gauge[D_gauge.split == "train"], D_sat[D_sat.split == "train"]
    feats_s = feats + ["obs_age"]

    rows, keep = [], []
    for h in fx.HORIZONS:
        mdl = lambda: fx.models()["HistGB"]()
        mA = mdl().fit(tr_g.dropna(subset=[f"y{h}"])[feats], tr_g.dropna(subset=[f"y{h}"])[f"y{h}"])
        mB = mdl().fit(tr_s.dropna(subset=[f"y{h}"])[feats_s], tr_s.dropna(subset=[f"y{h}"])[f"y{h}"])
        for r in fx.RES:
            # A: gauge input (reference)
            Dg = D_gauge[(D_gauge.res == r)].loc[TEST_START:TEST_END].dropna(subset=[f"y{h}"])
            pA = fx.to_storage(Dg, mA.predict(Dg[feats]), h); yA = fx.truth(Dg, h)
            res = {"A gauge input": pd.Series(pA, Dg.index)}
            truth_ser = pd.Series(yA, Dg.index)
            # B / D: satellite input (S2 / S1), forecaster trained on all 4 with satellite-like inputs
            for tag, sensor in [("B S2 input", "S2"), ("D S1-Otsu input", "S1")]:
                T = test_frame(L, R, S, cfg, r, sensor).dropna(subset=[f"y{h}"])
                res[tag] = pd.Series(fx.to_storage(T, mB.predict(T[feats_s]), h), T.index)
                if sensor == "S2":
                    res["P S2 persistence"] = pd.Series(T.s_l0.values * T.cap.values, T.index)
            # E: error budget — real S2 observation DATES, but exact (gauge) storage at those dates
            S_perfect = S.assign(frac_LORO=S.frac)
            Te = test_frame(L, R, S_perfect, cfg, r, "S2").dropna(subset=[f"y{h}"])
            res["E S2 dates, exact storage"] = pd.Series(fx.to_storage(Te, mB.predict(Te[feats_s]), h), Te.index)
            # C: fully gauge-free — LORO forecaster + LORO rating + CHIRPS rain
            a_ = tr_s[(tr_s.res != r)].dropna(subset=[f"y{h}"])
            mC = mdl().fit(a_[feats_s], a_[f"y{h}"])
            Tc = test_frame(L, chirps, S, cfg, r, "S2").dropna(subset=[f"y{h}"])
            res["C S2 + CHIRPS, unseen reservoir"] = pd.Series(fx.to_storage(Tc, mC.predict(Tc[feats_s]), h), Tc.index)
            # score all on the common dates
            common = truth_ser.index
            for s in res.values():
                common = common.intersection(s.index)
            cap = cfg[r]["capacity"]
            for tag, s in res.items():
                m = regression_metrics(truth_ser[common] / cap, s[common] / cap)
                rows.append({"horizon": h, "reservoir": r, "config": tag, "n": len(common), **m})
            keep.append(pd.DataFrame({"date": common, "reservoir": r, "h": h, "truth": truth_ser[common].values,
                                      **{k: v[common].values for k, v in res.items()}}))
    M = pd.DataFrame(rows); M.to_csv(out / "E5_satellite_forecast_metrics.csv", index=False)
    P = pd.concat(keep); P.to_csv(out / "E5_predictions.csv", index=False)
    # pooled over reliable-gauge reservoirs
    pooled = []
    for (h, tag), g in P.melt(id_vars=["date", "reservoir", "h", "truth"], var_name="config", value_name="pred").groupby(["h", "config"]):
        g = g[~g.reservoir.isin(["CHOLAVARAM"])]
        cap = g.reservoir.map(lambda x: cfg[x]["capacity"])
        pooled.append({"horizon": h, "config": tag, **regression_metrics(g.truth / cap, g.pred / cap)})
    PO = pd.DataFrame(pooled); PO.to_csv(out / "E5_pooled_excl_cholavaram.csv", index=False)
    json.dump(chirps_corr, open(out / "chirps_vs_gauge_monthly_corr.json", "w"))
    pd.set_option("display.width", 200)
    print(PO.pivot(index="config", columns="horizon", values="NSE").round(3))
    print(PO.pivot(index="config", columns="horizon", values="RMSE").mul(100).round(1).add_suffix(" RMSE%"))
    print(M.pivot_table(index=["reservoir", "config"], columns="horizon", values="NSE").round(3))
    figure(out, P, cfg)


def figure(out, P, cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    h = 15
    f, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for i, r in enumerate(["POONDI", "REDHILLS", "CHEMBARAMBAKKAM"]):
        g = P[(P.reservoir == r) & (P.h == h)].sort_values("date")
        cap = cfg[r]["capacity"]; t = pd.to_datetime(g.date) + pd.Timedelta(days=h)
        ax[i].plot(t, g.truth / cap * 100, "k", lw=1.1, label="Gauge (truth)")
        ax[i].plot(t, g["A gauge input"] / cap * 100, color="#888", lw=.8, label="A: gauge input")
        ax[i].plot(t, g["B S2 input"] / cap * 100, color="#1f5aa6", lw=.9, label="B: real S2 input")
        ax[i].plot(t, g["C S2 + CHIRPS, unseen reservoir"] / cap * 100, color="#c0392b", lw=.9,
                   label="C: fully gauge-free")
        ax[i].set_ylabel(f"{r.title()}\n% capacity")
    ax[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    ax[0].set_title(f"{h}-day-ahead storage forecasts driven by real Sentinel-2 observations (2019-2020)")
    plt.savefig(out / "figures/S3_satellite_driven_forecasts.png"); plt.close()


if __name__ == "__main__":
    main()
