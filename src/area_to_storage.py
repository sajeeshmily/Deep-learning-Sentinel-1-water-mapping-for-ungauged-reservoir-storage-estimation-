"""
area_to_storage.py — convert satellite water area (km^2) into storage (or level), three ways:

  1. GAUGED rating curve  : fit S = a + b * A^c on that reservoir's own gauge overlap.
  2. REGIONAL (ungauged)  : leave-one-reservoir-out. Fit one dimensionless curve
                            S/Smax = f(A/Amax) on the OTHER reservoirs, apply to the held-out
                            one using only its public design capacity Smax and its JRC
                            max-extent area Amax. No gauge data from the target is used.
  3. DEM hypsometry       : integrate area-elevation from Copernicus DEM (GLO-30 sees the
                            water surface, not the bed, so the curve below the DEM-epoch
                            surface is extrapolated with a fitted power law). Fully gauge-free.

Usage
  python src/area_to_storage.py --area data/area_timeseries.csv --gauge data/gauge_long.csv \
      --config configs/reservoirs.json --out outputs/storage
  (--dem-dir data/dem to also run method 3; needs rasterio)

Gauge file (long format): date,reservoir,storage   (storage in the units of the config)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import regression_metrics


def power(a_, a, b, c):
    return a + b * np.power(np.clip(a_, 1e-9, None), c)


def fit_power(x, y):
    p0 = [np.min(y), (np.max(y) - np.min(y)) / max(np.max(x) ** 1.3, 1e-9), 1.3]
    try:
        p, _ = curve_fit(power, x, y, p0=p0, bounds=([-np.inf, 0, 0.3], [np.inf, np.inf, 4.0]),
                         maxfev=20000)
    except RuntimeError:
        b, a = np.polyfit(x, y, 1); p = [a, max(b, 0), 1.0]
    return np.asarray(p)


def match_gauge(area, gauge, tol_days=1):
    """Nearest gauge reading within tol_days of each satellite date."""
    out = []
    for res, g in gauge.groupby("reservoir"):
        a = area[area.reservoir == res].sort_values("date")
        if a.empty:
            continue
        m = pd.merge_asof(a, g.sort_values("date")[["date", "storage"]], on="date",
                          direction="nearest", tolerance=pd.Timedelta(days=tol_days))
        out.append(m.dropna(subset=["storage"]))
    return pd.concat(out) if out else pd.DataFrame()


def time_blocked_cv(x, y, dates, k=5):
    """k contiguous time blocks; fit on k-1, predict the held-out block (no leakage)."""
    order = np.argsort(dates.values); x, y = x[order], y[order]
    pred = np.full_like(y, np.nan, dtype=float)
    for idx in np.array_split(np.arange(len(x)), k):
        tr = np.setdiff1d(np.arange(len(x)), idx)
        if len(tr) < 5:
            continue
        pred[idx] = power(x[idx], *fit_power(x[tr], y[tr]))
    return y, pred


# ----------------------------------------------------------------------------- DEM method
def hypsometry(dem, inside, px_area_km2, n=200):
    """Area-elevation-volume from a DEM inside the reservoir mask.
    Returns dict with z, A(z) km2, V(z) (km2*m = 1e6 m3 = 1 MCM), z_surface (DEM-epoch water level)."""
    z = dem[inside & np.isfinite(dem)]
    # DEM-epoch water surface ~ the most common low elevation (flat water plane)
    # The water plane is flat, so it is the single most frequent elevation (0.25 m bins).
    vals, counts = np.unique(np.round(z * 4) / 4, return_counts=True)
    z_surf = float(vals[np.argmax(counts)])
    zz = np.linspace(z_surf, np.percentile(z, 99.5), n)
    A = np.array([(z <= v).sum() * px_area_km2 for v in zz])
    # extrapolate below surface: A(z) = k * (z - z_bed)^m, fitted on the lowest 40 % of the curve
    sel = slice(1, max(int(0.4 * n), 5))
    def f(zv, k, zb, m): return k * np.power(np.clip(zv - zb, 1e-6, None), m)
    try:
        (k, zb, m), _ = curve_fit(f, zz[sel], A[sel], p0=[A[sel].max(), z_surf - 5, 1.5],
                                  bounds=([0, z_surf - 40, 0.5], [np.inf, z_surf - 0.1, 4]), maxfev=20000)
    except RuntimeError:
        k, zb, m = A[1] / 5 ** 1.5, z_surf - 5, 1.5
    z_full = np.concatenate([np.linspace(zb, z_surf, 50, endpoint=False), zz])
    A_full = np.concatenate([f(z_full[:50], k, zb, m), A])
    V_full = np.concatenate([[0], np.cumsum(0.5 * (A_full[1:] + A_full[:-1]) * np.diff(z_full))])
    return {"z": z_full, "A": A_full, "V_mcm": V_full, "z_surface": float(z_surf), "z_bed": float(zb)}


def area_to_volume_dem(area_km2, h):
    A, V, z = h["A"], h["V_mcm"], h["z"]
    order = np.argsort(A)
    return np.interp(area_km2, A[order], V[order]), np.interp(area_km2, A[order], z[order])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True)
    ap.add_argument("--gauge", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dem-dir")
    ap.add_argument("--area-col", default="area_km2")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    area = pd.read_csv(a.area, parse_dates=["date"]).rename(columns={a.area_col: "A"})
    gauge = pd.read_csv(a.gauge, parse_dates=["date"])
    # fall back to the largest observed satellite area when max_area_km2 is not set
    for r, g in area.groupby("reservoir"):
        cfg.setdefault(r, {})
        if not cfg[r].get("max_area_km2"):
            cfg[r]["max_area_km2"] = float(g.A.max())
    m = match_gauge(area, gauge)
    res_list = sorted(m.reservoir.unique())
    rows, series = [], []

    for res in res_list:
        d = m[m.reservoir == res]
        # 1. gauged, time-blocked CV
        y, p = time_blocked_cv(d.A.values, d.storage.values, d.date)
        ok = np.isfinite(p)
        rows.append({"reservoir": res, "method": "gauged_rating", "n": int(ok.sum()),
                     **regression_metrics(y[ok], p[ok])})
        # 2. regional leave-one-reservoir-out
        others = m[m.reservoir != res]
        if others.reservoir.nunique() >= 1:
            xo = np.concatenate([others[others.reservoir == r].A.values / cfg[r]["max_area_km2"] for r in others.reservoir.unique()])
            yo = np.concatenate([others[others.reservoir == r].storage.values / cfg[r]["capacity"] for r in others.reservoir.unique()])
            pr = fit_power(xo, yo)
            pred = power(d.A.values / cfg[res]["max_area_km2"], *pr) * cfg[res]["capacity"]
            rows.append({"reservoir": res, "method": "regional_LORO", "n": len(d),
                         **regression_metrics(d.storage.values, pred)})
            series.append(pd.DataFrame({"date": d.date.values, "reservoir": res, "A": d.A.values,
                                        "storage_obs": d.storage.values,
                                        "storage_regional": pred}))
        # 3. DEM (optional)
        if a.dem_dir:
            import rasterio, rasterio.features
            from rasterio.warp import transform_geom
            with rasterio.open(Path(a.dem_dir) / f"DEM_{res}.tif") as src:
                dem = src.read(1).astype(float); prof = src.profile
            gj = json.loads((Path(a.dem_dir) / f"mask_{res}.geojson").read_text())
            g = [transform_geom("EPSG:4326", prof["crs"], f["geometry"]) for f in gj["features"]]
            inside = rasterio.features.geometry_mask(g, dem.shape, prof["transform"], invert=True)
            h = hypsometry(dem, inside, abs(prof["transform"].a * prof["transform"].e) / 1e6)
            v_mcm, _ = area_to_volume_dem(d.A.values, h)
            v = v_mcm / cfg[res].get("unit_to_mcm", 1.0)
            # DEM volume is relative to an extrapolated bed: report raw and offset-corrected skill
            rows.append({"reservoir": res, "method": "dem_hypsometry_raw", "n": len(d),
                         **regression_metrics(d.storage.values, v)})
            np.save(out / f"hypsometry_{res}.npy", h, allow_pickle=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "area_to_storage_metrics.csv", index=False)
    if series:
        pd.concat(series).to_csv(out / "storage_from_area.csv", index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
