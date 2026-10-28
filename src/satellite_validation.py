"""
satellite_validation.py — validate satellite water area against gauge storage (real data).

Inputs : data/gee/baseline_areas_<RES>.csv (Earth Engine export: S2 MNDWI and S1 VV-Otsu areas)
         data/chennai_reservoir_levels.csv (gauge storage, QC'd with the same spike filter)
Outputs: outputs/satellite/  (clean series, metrics, figures)

Cleaning rules (documented for the paper):
  S2  * keep only tiles that fully cover the reservoir (coverage_frac > 0.95 in the newer export;
        in the first export, tile max area >= 0.8 x the best tile's max)
      * per date: largest area among full-cover tiles with cloud fraction < CLOUD_MAX
      * S2 dates before 2019 are sparse: the L2A (SCL cloud mask) archive over India starts late 2018
  S1  * keep the dominant relative orbit (track 92, descending, ~00:30 UTC)
      * from 2020-03 each pass is split across two consecutive frames: per-date SUM of frame areas
        (upper bound; small along-track overlap)
  Gauge: CHOLAVARAM is flagged — storage stays flat at ~50-80 mcft for months while both sensors
         show the water spread changing, so it is excluded from the regional rating's TRAINING set.

  python src/satellite_validation.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from area_to_storage import fit_power, power, time_blocked_cv
from forecast_experiments import RES, load, qc_spikes
from metrics import regression_metrics

CLOUD_MAX = 0.10
FLAGGED_GAUGE = {"CHOLAVARAM"}


def s1_track(ids):
    p = ids.str.split("_")
    sat, absorb = p.str[0], p.str[6].astype(int)
    return np.where(sat == "S1A", (absorb - 73) % 175 + 1, (absorb - 27) % 175 + 1)


def clean_s2(d, cloud_max=CLOUD_MAX):
    s2 = d[d.sensor == "S2"].copy()
    s2["tile"] = s2["id"].str[-6:]
    clear = s2[s2.cloud_frac < cloud_max]
    if "coverage_frac" in clear:   # newer export: footprint coverage measured directly
        clear = clear[clear.coverage_frac > 0.95]
        full = sorted(clear.tile.unique())
    else:                          # first export: infer full-cover tiles from their max area
        tmax = clear.groupby("tile").area_km2.max()
        full = tmax[tmax >= 0.8 * tmax.max()].index
    out = clear[clear.tile.isin(full)].groupby("date").agg(area_km2=("area_km2", "max"),
                                                           cloud_frac=("cloud_frac", "min"))
    return out.reset_index(), list(full)


def clean_s1(d):
    s1 = d[d.sensor == "S1"].copy()
    if "track" not in s1 or s1["track"].isna().all():   # first export: derive track from scene id
        s1["track"] = s1_track(s1["id"])
    # (newer export already mosaics same-day frames, so the per-date sum below is a no-op)
    main = s1.track.value_counts().idxmax()
    s1 = s1[s1.track == main]
    return s1.groupby("date").agg(area_km2=("area_km2", "sum"), n_frames=("id", "count")).reset_index(), int(main)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/satellite"; (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "configs/reservoirs.json").read_text())
    L, _ = load(root / "data"); L = qc_spikes(L, cfg)

    series, info = [], {}
    for r in RES:
        d = pd.read_csv(root / f"data/gee/baseline_areas_{r}.csv", parse_dates=["date"])
        s2, tiles = clean_s2(d); s1, trk = clean_s1(d)
        info[r] = {"s2_full_tiles": tiles, "s1_track": trk}
        for sensor, s in [("S2", s2), ("S1", s1)]:
            s = s.assign(reservoir=r, sensor=sensor)
            s["storage"] = L[r].reindex(s.date).values
            s["frac"] = s.storage / cfg[r]["capacity"]
            series.append(s)
    S = pd.concat(series, ignore_index=True)
    S = S[S.date <= L.index.max()].dropna(subset=["storage"])
    S.to_csv(out / "satellite_areas_clean.csv", index=False)

    # ------------------------------------------------------------------ area vs storage skill
    rows = []
    for sensor in ["S2", "S1"]:
        for r in RES:
            g = S[(S.sensor == sensor) & (S.reservoir == r)].sort_values("date")
            cap = cfg[r]["capacity"]
            row = {"sensor": sensor, "reservoir": r, "n": len(g),
                   "pearson_r": stats.pearsonr(g.area_km2, g.storage)[0],
                   "spearman_rho": stats.spearmanr(g.area_km2, g.storage)[0]}
            y, p = time_blocked_cv(g.area_km2.values, g.storage.values, g.date)
            ok = np.isfinite(p)
            m = regression_metrics(y[ok] / cap, p[ok] / cap)
            row.update({f"gauged_{k}": v for k, v in m.items() if k in ("RMSE", "NSE", "KGE")})
            # regional leave-one-reservoir-out: normalise by each reservoir's max satellite area
            # (gauge-free) and capacity; train only on unflagged OTHER reservoirs
            tr = S[(S.sensor == sensor) & (S.reservoir != r) & (~S.reservoir.isin(FLAGGED_GAUGE))]
            amax = S[S.sensor == sensor].groupby("reservoir").area_km2.max()
            pr = fit_power((tr.area_km2 / tr.reservoir.map(amax)).values, tr.frac.values)
            pred = np.clip(power(g.area_km2.values / amax[r], *pr), 0, 1.05)
            m = regression_metrics(g.frac.values, pred)
            row.update({f"LORO_{k}": v for k, v in m.items() if k in ("RMSE", "NSE", "KGE", "PBIAS_%")})
            S.loc[g.index, "frac_LORO"] = pred
            rows.append(row)
    M = pd.DataFrame(rows); M.to_csv(out / "area_vs_storage_metrics.csv", index=False)

    # ------------------------------------------------------------------ dry-bed false water (S1)
    dry = []
    for r in RES:
        a2 = S[(S.sensor == "S2") & (S.reservoir == r)]; a1 = S[(S.sensor == "S1") & (S.reservoir == r)]
        for sensor, g in [("S2", a2), ("S1", a1)]:
            amax = g.area_km2.max()
            dry.append({"reservoir": r, "sensor": sensor,
                        "n_dry": int((g.frac < 0.05).sum()),
                        "median_area_when_storage<5%cap_km2": g[g.frac < 0.05].area_km2.median(),
                        "as_%_of_max_area": 100 * g[g.frac < 0.05].area_km2.median() / amax})
    DRY = pd.DataFrame(dry); DRY.to_csv(out / "dry_bed_false_water.csv", index=False)

    # ------------------------------------------------------------------ observation availability
    S["month"] = S.date.dt.month
    S["season"] = np.where(S.month >= 10, "NE (Oct-Dec)", np.where(S.month >= 6, "SW (Jun-Sep)", "Dry (Jan-May)"))
    win = S[S.date >= "2019-01-01"]  # S2 archive complete from 2019
    gaps = []
    for (sensor, r), g in win.groupby(["sensor", "reservoir"]):
        dts = g.date.sort_values()
        gg = dts.diff().dt.days
        for sn in ["Dry (Jan-May)", "SW (Jun-Sep)", "NE (Oct-Dec)"]:
            sel = g.season.values == sn
            gaps.append({"sensor": sensor, "reservoir": r, "season": sn,
                         "obs_per_month": sel.sum() / ({"Dry (Jan-May)": 5, "SW (Jun-Sep)": 4, "NE (Oct-Dec)": 3}[sn] * 2),
                         "max_gap_days": gg[sel].max()})
    G = pd.DataFrame(gaps); G.to_csv(out / "observation_availability_2019_2020.csv", index=False)

    S.to_csv(out / "satellite_areas_clean.csv", index=False)
    json.dump(info, open(out / "cleaning_info.json", "w"), indent=2)
    figures(out, S, L, cfg)
    pd.set_option("display.width", 200)
    print(json.dumps(info))
    print(M.round(3).to_string(index=False)); print(DRY.round(2).to_string(index=False))
    print(G.groupby(["sensor", "season"])[["obs_per_month", "max_gap_days"]].mean().round(1))


def figures(out, S, L, cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    c = {"S2": "#1f5aa6", "S1": "#c0392b"}
    # scatter area vs storage
    f, ax = plt.subplots(2, 4, figsize=(11, 5))
    for j, r in enumerate(RES):
        for i, sensor in enumerate(["S2", "S1"]):
            g = S[(S.sensor == sensor) & (S.reservoir == r)]
            ax[i, j].scatter(g.area_km2, g.frac * 100, s=7, color=c[sensor], alpha=.6)
            rho = stats.spearmanr(g.area_km2, g.storage)[0]
            ax[i, j].set_title(f"{r.title()} — {sensor}  (ρ={rho:.2f})", fontsize=8)
            ax[i, j].set_xlabel("Water area (km²)"); ax[i, j].set_ylabel("Gauge storage (% cap)")
    plt.tight_layout(); plt.savefig(out / "figures/S1_area_vs_storage.png"); plt.close()
    # time series of LORO satellite storage
    f, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for i, r in enumerate(RES):
        cap = cfg[r]["capacity"]
        ax[i].plot(L.loc["2017":"2020"].index, L.loc["2017":"2020", r] / cap * 100, "k", lw=1, label="Gauge")
        for sensor in ["S2", "S1"]:
            g = S[(S.sensor == sensor) & (S.reservoir == r)]
            ax[i].plot(g.date, g.frac_LORO * 100, "o", ms=2.5, color=c[sensor],
                       label=f"{sensor} → storage (ungauged rating)")
        ax[i].set_ylabel(f"{r.title()}\n% capacity"); ax[i].set_ylim(-5, 110)
    ax[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax[0].set_title("Satellite-derived storage using a rating curve fitted only on OTHER reservoirs")
    plt.savefig(out / "figures/S2_satellite_storage_timeseries.png"); plt.close()


if __name__ == "__main__":
    main()
