"""
stream_inference.py — run the trained U-Nets on the Chennai reservoirs WITHOUT exporting images.

Each scene is fetched from Earth Engine straight into memory (ee.data.computePixels) on a fixed
10 m UTM grid, the models are applied, only water areas are kept, and the image is discarded.
Nothing is written to Google Drive.

For every scene it records, on identical pixels:
  U-Nets      : s2 (cloud-free pixels only), s1, fused        -> model = s2 / s1 / fused
  baselines   : MNDWI>0, NDWI>0 (cloud-free pixels), S1 VV Otsu -> model = mndwi / ndwi / s1_otsu
Columns: date, reservoir, model, input, area_km2, area_std_km2, cloud_frac, clear_area_km2

Resumable: one CSV per reservoir in --out; finished reservoirs are skipped on re-run.

  python src/stream_inference.py --project reservoir-forecast-2026 --models models --out out_areas
"""
import argparse, json, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from area_timeseries import CLOUD_PROB_THR, pair_scenes, run_scene, to_full_s2
from water_seg_common import DEFAULT_S2_USE, S2_BANDS, otsu_threshold, prep_s1

START, END = "2017-01-01", "2020-12-31"
CRS, SCALE = "EPSG:32644", 10
MAX_TILE_CLOUD = 70
BOX_BUFFER_M = 6000
RESERVOIRS = {
    "POONDI": (79.860, 13.190), "CHOLAVARAM": (80.150, 13.220),
    "REDHILLS": (80.180, 13.160), "CHEMBARAMBAKKAM": (80.060, 13.010),
}


# ----------------------------------------------------------------------------- baselines on same pixels
def baseline_rows(s2, s1, inside, px_km2):
    rows = []
    if s2 is not None:
        cloud = (s2[13] > CLOUD_PROB_THR) | (s2[:13].sum(0) == 0)
        valid = inside & ~cloud
        g, n, sw = (s2[S2_BANDS.index(b)].astype(np.float32) for b in ("B3", "B8", "B11"))
        mndwi = (g - sw) / np.maximum(g + sw, 1e-6) > 0
        ndwi = (g - n) / np.maximum(g + n, 1e-6) > 0
        cf = float(cloud[inside].mean()); ca = float(valid.sum() * px_km2)
        for name, w in (("mndwi", mndwi), ("ndwi", ndwi)):
            rows.append({"model": name, "input": "S2", "area_km2": float((w & valid).sum() * px_km2),
                         "area_std_km2": 0.0, "cloud_frac": cf, "clear_area_km2": ca})
    if s1 is not None:
        vv = prep_s1(s1)[0]
        t = otsu_threshold(vv)  # histogram over the whole fetched rectangle (as in Phase 1)
        t = float(np.clip(t, -22, -12)) if np.isfinite(t) else -16.0
        rows.append({"model": "s1_otsu", "input": "S1 only", "area_km2": float(((vv < t) & inside).sum() * px_km2),
                     "area_std_km2": 0.0, "cloud_frac": np.nan, "clear_area_km2": np.nan})
    return rows


# ----------------------------------------------------------------------------- Earth Engine side
def ee_setup(project):
    import ee
    ee.Initialize(project=project)
    return ee


def reservoir_grid(ee, lon, lat):
    """Reservoir polygon (JRC max extent + 300 m) and a fixed 10 m UTM grid around it."""
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    pt = ee.Geometry.Point([lon, lat]); box = pt.buffer(BOX_BUFFER_M).bounds()
    lab = jrc.select("max_extent").eq(1).clip(box).selfMask().connectedComponents(ee.Kernel.plus(1), 1024).select("labels")
    lbl = lab.reduceRegion(reducer=ee.Reducer.mode(), geometry=pt.buffer(300), scale=30, maxPixels=1e9).get("labels")
    poly = lab.eq(ee.Number(lbl)).selfMask().reduceToVectors(geometry=box, scale=30, maxPixels=1e10).geometry().buffer(300)
    proj = ee.Projection(CRS)
    coords = poly.transform(proj, 1).bounds(1, proj).coordinates().getInfo()[0]
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    if max(abs(v) for v in xs + ys) < 1000:  # got degrees, not metres: projection step failed
        raise RuntimeError(f"grid bounds not in {CRS} metres: {coords}")
    x0, y1 = np.floor(min(xs) / SCALE) * SCALE, np.ceil(max(ys) / SCALE) * SCALE
    W, H = int(np.ceil((max(xs) - x0) / SCALE)), int(np.ceil((y1 - min(ys)) / SCALE))
    grid = {"dimensions": {"width": W, "height": H},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": float(x0),
                                "shearY": 0, "scaleY": -SCALE, "translateY": float(y1)},
            "crsCode": CRS}
    region = poly.bounds()
    return poly, region, grid


def fetch(ee, img, bands, grid, retries=5):
    for k in range(retries):
        try:
            arr = ee.data.computePixels({"expression": img.select(bands), "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            return np.stack([arr[b].astype(np.float32) for b in bands])
        except Exception as e:  # rate limits / transient errors
            if k == retries - 1:
                raise
            time.sleep(2 ** k + np.random.rand())


def list_scenes(ee, region):
    l1c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region).filterDate(START, END) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_TILE_CLOUD))
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(region).filterDate(START, END)
    s2 = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
        l1c, prob, ee.Filter.equals(leftField="system:index", rightField="system:index")))
    s2 = s2.map(lambda im: im.set("covers", im.geometry().contains(region, ee.ErrorMargin(50)))) \
        .filter(ee.Filter.eq("covers", True))
    best = {}
    for sid, cc in s2.reduceColumns(ee.Reducer.toList(2), ["system:index", "CLOUDY_PIXEL_PERCENTAGE"]).get("list").getInfo():
        if sid[:8] not in best or cc < best[sid[:8]][1]:
            best[sid[:8]] = (sid, cc)
    s2_rows = [{"sensor": "S2", "time": datetime.strptime(v[0][:15], "%Y%m%dT%H%M%S"), "path": v[0]} for v in best.values()]

    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(START, END) \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    meta = s1.reduceColumns(ee.Reducer.toList(3), ["system:index", "system:time_start", "relativeOrbitNumber_start"]) \
        .get("list").getInfo()
    groups = defaultdict(list)
    for sid, t_ms, trk in meta:
        groups[(datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime("%Y%m%d"), int(trk))].append(sid)
    main = Counter(trk for _, trk in groups).most_common(1)[0][0]
    s1_rows = [{"sensor": "S1", "time": datetime.strptime(day, "%Y%m%d"), "path": tuple(ids)}
               for (day, trk), ids in groups.items() if trk == main]
    return s2_rows, s1_rows, main


def scene_images(ee, s2r, s1r):
    s2_img = s1_img = None
    if s2r:
        cp = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{s2r['path']}").select("probability")
        s2_img = ee.Image(f"COPERNICUS/S2_HARMONIZED/{s2r['path']}").select(DEFAULT_S2_USE).addBands(cp).unmask(0)
    if s1r:
        s1_img = ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{x}") for x in s1r["path"]]).mosaic() \
            .select(["VV", "VH"]).unmask(-9999)
    return s2_img, s1_img


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reservoirs", default=",".join(RESERVOIRS))
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--pair-days", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="parallel Earth Engine downloads")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes per reservoir (quick test)")
    a = ap.parse_args()

    import pandas as pd, torch
    from area_timeseries import load_models, torch_predict
    ee = ee_setup(a.project)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    models = load_models(a.models, ["s2", "s1", "fused"], device)
    predict = torch_predict(device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for res in a.reservoirs.split(","):
        dst = out / f"areas_{res}.csv"
        if dst.exists() and not a.limit:
            print("done already:", res); continue
        t0 = time.time()
        lon, lat = RESERVOIRS[res]
        poly, region, grid = reservoir_grid(ee, lon, lat)
        mask_img = ee.Image.constant(0).paint(ee.FeatureCollection([ee.Feature(poly)]), 1).rename("m")
        inside = fetch(ee, mask_img, ["m"], grid)[0] > 0.5
        px_km2 = SCALE * SCALE / 1e6
        s2_rows, s1_rows, trk = list_scenes(ee, region)
        rows = [dict(r, res=res) for r in s2_rows + s1_rows]
        pairs = pair_scenes(rows, res, a.pair_days)
        if a.limit:
            pairs = pairs[:a.limit]
        print(f"{res}: grid {grid['dimensions']}, mask {inside.sum() * px_km2:.1f} km2, "
              f"{len(s2_rows)} S2 + {len(s1_rows)} S1 (track {trk}) -> {len(pairs)} scenes", flush=True)

        def load(pair):
            s2r, s1r = pair
            s2_img, s1_img = scene_images(ee, s2r, s1r)
            s2 = fetch(ee, s2_img, DEFAULT_S2_USE + ["probability"], grid) if s2_img is not None else None
            s1 = fetch(ee, s1_img, ["VV", "VH"], grid) if s1_img is not None else None
            if s1 is not None:
                s1 = np.where(s1 < -5000, np.nan, s1)
            return pair, s2, s1

        results, failed = [], 0
        window = 2 * a.workers  # bounded prefetch: at most this many scenes held in memory
        with ThreadPoolExecutor(a.workers) as pool:
            futs = {}
            for i in range(len(pairs)):
                while len(futs) < window and (i + len(futs)) < len(pairs):
                    j = i + len(futs); futs[j] = pool.submit(load, pairs[j])
                f = futs.pop(i)
                try:
                    (s2r, s1r), s2, s1 = f.result()
                    s2 = to_full_s2(s2) if s2 is not None else None
                except Exception as e:
                    failed += 1; print("  fetch failed:", repr(e)[:200], flush=True); continue
                t = (s2r or s1r)["time"].strftime("%Y-%m-%d")
                for r in run_scene(models, s2, s1, inside, px_km2, a.mc, predict) + baseline_rows(s2, s1, inside, px_km2):
                    results.append({"date": t, "reservoir": res, **r})
                if (i + 1) % 25 == 0:
                    print(f"  {res}: {i + 1}/{len(pairs)} scenes ({(time.time() - t0) / 60:.1f} min)", flush=True)
        df = pd.DataFrame(results)
        if not a.limit:
            df.to_csv(dst, index=False)
        print(f"{res}: {len(df)} rows, {failed} failed scenes, {(time.time() - t0) / 60:.1f} min", flush=True)
        if a.limit:
            print(df.groupby("model").area_km2.describe().round(2))

    if not a.limit:
        allr = pd.concat([pd.read_csv(f) for f in sorted(out.glob("areas_*.csv"))])
        allr.to_csv(out / "area_timeseries_all.csv", index=False)
        print("wrote", out / "area_timeseries_all.csv", len(allr), "rows")


if __name__ == "__main__":
    main()
