"""
area_timeseries.py — run the trained U-Nets over the Earth Engine image stacks and build water-area
time series for each reservoir, with EVERY model on every usable scene, so the choice of model
(S2-only, S1-only, fused) is decided on the reservoir data rather than assumed.

Inputs (Phase 2 of gee/01_export_reservoir_data_colab.py, copied from Google Drive):
  stacks/S2_<RES>_<YYYYMMDDTHHMMSS>_<TILE>.tif   B2,B3,B4,B8,B11,B12 + s2cloudless probability (compact)
                                          (the older 14-band layout is also accepted)
  stacks/S1_<RES>_<YYYYMMDDTHHMMSS>.tif   VV, VH as int16 centi-dB (-9999 = no data), one mosaic per day/track
  masks/mask_<RES>.geojson                (Phase 1)
  models/{s2,s1,fused}_resnet18_s0/{best.pt,norm.json}   (Kaggle run)

Run on Kaggle/Colab (GPU optional; CPU works, slower):
  pip install -q segmentation-models-pytorch rasterio
  python src/area_timeseries.py --stacks data/stacks --masks data/gee --models models \
      --out data/area_timeseries_unet.csv

Output: one row per (reservoir, date, model) with
  area_km2       water area inside the reservoir mask (S2 models: cloud-free pixels only)
  area_std_km2   MC-dropout spread of the area
  cloud_frac     cloud fraction over the reservoir (S2 scenes; 1.0 when no S2 scene)
  clear_area_km2 reservoir area that was cloud-free (to judge how complete an S2 estimate is)
Pairing: an S2 scene and an S1 scene within --pair-days are fused; S1 dates without an S2
partner are run with the optical channels zeroed (as in robustness training).
"""
import argparse, json, re, sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water_seg_common import DEFAULT_S2_USE, S2_BANDS, prep_s1, prep_s2

CLOUD_PROB_THR = 50  # s2cloudless probability (%)
NAME_RE = re.compile(r"(S1|S2)_([A-Z]+)_(\d{8}T\d{6})")


def scan(stacks):
    rows = []
    for f in sorted(Path(stacks).glob("*.tif")):
        m = NAME_RE.search(f.stem)
        if m:
            rows.append({"sensor": m.group(1), "res": m.group(2),
                         "time": datetime.strptime(m.group(3), "%Y%m%dT%H%M%S"), "path": f})
    return rows


def pair_scenes(rows, res, pair_days):
    """List of (s2_row or None, s1_row or None). Each S1 scene is used at most once."""
    s2 = sorted([r for r in rows if r["res"] == res and r["sensor"] == "S2"], key=lambda r: r["time"])
    s1 = sorted([r for r in rows if r["res"] == res and r["sensor"] == "S1"], key=lambda r: r["time"])
    used, items = set(), []
    for a in s2:
        cand = [b for b in s1 if b["path"] not in used and abs((b["time"] - a["time"]).days) <= pair_days]
        best = min(cand, key=lambda b: abs((b["time"] - a["time"]).total_seconds()), default=None)
        if best is not None:
            used.add(best["path"])
        items.append((a, best))
    items += [(None, b) for b in s1 if b["path"] not in used]
    return sorted(items, key=lambda t: (t[0] or t[1])["time"])


def scene_inputs(s2, s1, cloud, s2_use):
    """Model inputs (un-normalised) for each model family, plus which pixels each may score."""
    out = {}
    if s2 is not None:
        out["s2"] = prep_s2(s2[:13], s2_use)
    if s1 is not None:
        out["s1"] = prep_s1(s1)
        opt = prep_s2(s2[:13], s2_use) if s2 is not None else None
        out["fused"] = (prep_s1(s1), opt)  # optical zeroed after normalising (see run_scene)
    return out


def run_scene(models, s2, s1, inside, px_km2, mc, predict):
    """models: {family: (norm, cfg, model)}; predict(model, x, mc) -> list of prob maps."""
    cloud = np.zeros(inside.shape, bool)
    if s2 is not None:
        cloud = (s2[13] > CLOUD_PROB_THR) | (s2[:13].sum(0) == 0)
    cloud_frac = float(cloud[inside].mean()) if s2 is not None else 1.0
    clear_area = float((inside & ~cloud).sum() * px_km2) if s2 is not None else 0.0
    rows = []
    for fam, (norm, cfg, model) in models.items():
        if fam == "s2" and s2 is not None:
            x = norm(prep_s2(s2[:13], cfg["s2_use"])).astype(np.float32)
            valid = inside & ~cloud
        elif fam == "s1" and s1 is not None:
            x = norm(prep_s1(s1)).astype(np.float32)
            valid = inside
        elif fam == "fused" and s1 is not None:
            opt = (prep_s2(s2[:13], cfg["s2_use"]) if s2 is not None
                   else np.zeros((len(cfg["s2_use"]),) + inside.shape, np.float32))
            x = norm(np.concatenate([prep_s1(s1), opt], 0)).astype(np.float32)
            x[2:, cloud if s2 is not None else np.ones(inside.shape, bool)] = 0.0
            valid = inside
        else:
            continue
        areas = [((p > 0.5) & valid).sum() * px_km2 for p in predict(model, x, mc)]
        rows.append({"model": fam, "area_km2": float(np.mean(areas)), "area_std_km2": float(np.std(areas)),
                     "cloud_frac": cloud_frac, "clear_area_km2": clear_area,
                     "input": "S1+S2" if (fam == "fused" and s2 is not None) else
                              ("S1 only" if fam in ("s1", "fused") else "S2")})
    return rows


def to_full_s2(arr):
    """Return the 14-band layout (13 L1C bands + cloud probability) from either export layout:
    full (14 bands) or compact (6 model bands + cloud probability = 7 bands). Unused bands are 0."""
    if arr.shape[0] == 14:
        return arr
    if arr.shape[0] != len(DEFAULT_S2_USE) + 1:
        raise ValueError(f"unexpected S2 band count {arr.shape[0]}")
    full = np.zeros((14,) + arr.shape[1:], np.float32)
    for k, b in enumerate(DEFAULT_S2_USE):
        full[S2_BANDS.index(b)] = arr[k]
    full[13] = arr[-1]
    return full


def to_db_s1(arr, dtype):
    """S1 exports are float32 dB (old) or int16 centi-dB with -9999 no-data (compact)."""
    if np.issubdtype(np.dtype(dtype), np.integer):
        arr = np.where(arr < -5000, np.nan, arr / 100.0)
    return arr.astype(np.float32)


def read_on_grid(path, ref_profile):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as src:
        out = np.zeros((src.count, ref_profile["height"], ref_profile["width"]), np.float32)
        for b in range(src.count):
            reproject(rasterio.band(src, b + 1), out[b], dst_transform=ref_profile["transform"],
                      dst_crs=ref_profile["crs"], resampling=Resampling.bilinear)
    return out


def load_models(models_dir, families, device):
    import torch
    from train_unet_sen1floods11 import Normaliser, torch_parts
    _, _, make_model, _ = torch_parts()
    out = {}
    for fam in families:
        run = Path(models_dir) / f"{fam}_resnet18_s0"
        if not (run / "best.pt").exists():
            print("skip (no weights):", run); continue
        cfg = json.loads((run / "norm.json").read_text())
        norm = Normaliser(np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32))
        m = make_model(len(cfg["mean"]), cfg["encoder"]).to(device)
        m.load_state_dict(torch.load(run / "best.pt", map_location=device)); m.eval()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout2d):
                mod.train()  # MC dropout
        out[fam] = (norm, cfg, m)
    return out


def torch_predict(device):
    import torch

    def predict(model, x, mc):
        H, W = x.shape[1:]
        xt = torch.from_numpy(x[None]).to(device)
        xt = torch.nn.functional.pad(xt, (0, (-W) % 32, 0, (-H) % 32), mode="reflect")
        with torch.no_grad():
            return [torch.sigmoid(model(xt))[0, 0, :H, :W].float().cpu().numpy() for _ in range(mc)]
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="s2,s1,fused")
    ap.add_argument("--pair-days", type=int, default=3)
    ap.add_argument("--mc", type=int, default=10)
    a = ap.parse_args()

    import pandas as pd, rasterio, rasterio.features, torch
    from rasterio.warp import transform_geom
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(a.models, a.families.split(","), device)
    predict = torch_predict(device)

    rows = scan(a.stacks); out = []
    for res in sorted({r["res"] for r in rows}):
        gj = json.loads((Path(a.masks) / f"mask_{res}.geojson").read_text())
        geoms = [f["geometry"] for f in gj["features"]]
        pairs = pair_scenes(rows, res, a.pair_days)
        print(res, len(pairs), "scenes", flush=True)
        for s2r, s1r in pairs:
            ref = (s2r or s1r)["path"]
            with rasterio.open(ref) as src:
                prof = src.profile
            g = [transform_geom("EPSG:4326", prof["crs"], gg) for gg in geoms]
            inside = rasterio.features.geometry_mask(g, (prof["height"], prof["width"]), prof["transform"], invert=True)
            px_km2 = abs(prof["transform"].a * prof["transform"].e) / 1e6
            s2 = to_full_s2(read_on_grid(s2r["path"], prof)) if s2r else None
            s1 = None
            if s1r:
                with rasterio.open(s1r["path"]) as src:
                    s1_dtype = src.dtypes[0]
                s1 = to_db_s1(read_on_grid(s1r["path"], prof), s1_dtype)
            t = (s2r or s1r)["time"]
            for r in run_scene(models, s2, s1, inside, px_km2, a.mc, predict):
                out.append({"date": t.strftime("%Y-%m-%d"), "reservoir": res, **r})
    df = pd.DataFrame(out).sort_values(["reservoir", "date", "model"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")
    print(df.groupby(["reservoir", "model"]).size().unstack())


if __name__ == "__main__":
    main()
