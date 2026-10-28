"""
train_unet_sen1floods11.py — lightweight, cloud-robust SAR-optical U-Net for water segmentation.

Designed for a free Kaggle/Colab GPU (T4/P100, 16 GB). One run ~ 1-2 h at 256x256 crops.

Contributions it supports (paper Section 3.1):
  * fused S1+S2 input with MODALITY DROPOUT and SYNTHETIC CLOUD masking during training,
    so the model degrades gracefully when optical data is cloudy or missing;
  * MC-dropout uncertainty maps (mean prob + std), evaluated with ECE and error-AUROC;
  * robustness curve: IoU vs. cloud cover 0..100 %.

Data: Sen1Floods11 v1.1 hand-labelled split (public, gs://sen1floods11).
  Download on Kaggle/Colab (internet ON):
     pip install -q segmentation-models-pytorch rasterio google-cloud-storage
     python src/train_unet_sen1floods11.py --download --root /kaggle/working/sen1floods11
  Then train:
     python src/train_unet_sen1floods11.py --root /kaggle/working/sen1floods11 --mode fused --seed 0
     (repeat with --mode s2, --mode s1, --no-robust-train, and seeds 0,1,2 for the ablation table)

Outputs: outputs/seg/<run_name>/{best.pt, metrics.json, cloud_curve.csv, log.csv}
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water_seg_common import (DEFAULT_S2_USE, IGNORE, SegMeter, baseline_mndwi, baseline_ndwi,
                              baseline_s1_fixed, baseline_s1_otsu, expected_calibration_error,
                              prep_s1, prep_s2, synthetic_cloud_mask)

BUCKET = "sen1floods11"
PREFIX = "v1.1/"
SPLITS = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv",
          "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}


# ----------------------------------------------------------------------------- download
def download(root):
    """Anonymous download of the hand-labelled subset (~ 1.5 GB) + split files."""
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)
    wanted = [PREFIX + "splits/flood_handlabeled/",
              PREFIX + "data/flood_events/HandLabeled/S1Hand/",
              PREFIX + "data/flood_events/HandLabeled/S2Hand/",
              PREFIX + "data/flood_events/HandLabeled/LabelHand/"]
    n = 0
    for pre in wanted:
        for blob in client.list_blobs(bucket, prefix=pre):
            dst = Path(root) / blob.name[len(PREFIX):]
            if dst.exists() or blob.name.endswith("/"):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dst)); n += 1
    print(f"downloaded {n} files to {root}")


# ----------------------------------------------------------------------------- data
def read_split(root, split):
    f = Path(root) / "splits/flood_handlabeled" / SPLITS[split]
    ids = []
    with open(f) as fh:
        for row in csv.reader(fh):
            if row:
                ids.append(row[0].strip().replace("_S1Hand.tif", ""))
    return ids


def load_chip(root, cid):
    import rasterio
    d = Path(root) / "data/flood_events/HandLabeled"
    with rasterio.open(d / "S1Hand" / f"{cid}_S1Hand.tif") as r: s1 = r.read()
    with rasterio.open(d / "S2Hand" / f"{cid}_S2Hand.tif") as r: s2 = r.read()
    with rasterio.open(d / "LabelHand" / f"{cid}_LabelHand.tif") as r: lab = r.read(1)
    return s1, s2, lab.astype(np.int16)


class Normaliser:
    """Per-channel mean/std computed on the TRAIN split only (no test leakage)."""
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, arrays):
        x = np.concatenate([a.reshape(a.shape[0], -1) for a in arrays], axis=1)
        self.mean = x.mean(1).astype(np.float32); self.std = x.std(1).astype(np.float32) + 1e-6
        return self

    def __call__(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def build_input(s1, s2, mode, s2_use):
    if mode == "s1":
        return prep_s1(s1)
    if mode == "s2":
        return prep_s2(s2, s2_use)
    return np.concatenate([prep_s1(s1), prep_s2(s2, s2_use)], 0)


# ----------------------------------------------------------------------------- torch parts
def torch_parts():
    import torch, torch.nn as nn, torch.nn.functional as F
    import segmentation_models_pytorch as smp

    class ChipDS(torch.utils.data.Dataset):
        def __init__(self, chips, norm, mode, n_s1, train, crop, robust, seed):
            self.chips, self.norm, self.mode, self.n_s1 = chips, norm, mode, n_s1
            self.train, self.crop, self.robust = train, crop, robust
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.chips) * (4 if self.train else 1)  # 4 random crops/chip/epoch

        def __getitem__(self, i):
            x, y = self.chips[i % len(self.chips)]
            x = self.norm(x).astype(np.float32); y = y.copy()
            if self.train:
                H, W = y.shape; c = self.crop
                r0 = self.rng.integers(0, H - c + 1); c0 = self.rng.integers(0, W - c + 1)
                x = x[:, r0:r0+c, c0:c0+c]; y = y[r0:r0+c, c0:c0+c]
                if self.rng.random() < .5: x, y = x[:, :, ::-1], y[:, ::-1]
                if self.rng.random() < .5: x, y = x[:, ::-1], y[::-1]
                k = self.rng.integers(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
                if self.robust and self.mode == "fused":
                    u = self.rng.random()
                    if u < 0.25:   # optical completely missing (monsoon)
                        x[self.n_s1:] = 0.0          # 0 == train-set mean after standardising
                    elif u < 0.55: # partial cloud over optical
                        m = synthetic_cloud_mask(c, c, self.rng.uniform(.1, .9), self.rng)
                        x[self.n_s1:, m] = 0.0
                    elif u < 0.60: # SAR missing (rare)
                        x[:self.n_s1] = 0.0
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()

    def make_model(in_ch, encoder="resnet18", p_drop=0.2):
        m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=in_ch, classes=1)
        m.segmentation_head = nn.Sequential(nn.Dropout2d(p_drop), m.segmentation_head)
        return m

    def loss_fn(logits, y):
        valid = (y != IGNORE).float()
        t = (y == 1).float()
        logits = logits.squeeze(1)
        bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        p = torch.sigmoid(logits) * valid
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
        return bce + dice

    return torch, ChipDS, make_model, loss_fn


# ----------------------------------------------------------------------------- inference helpers
def predict_full(model, x, torch, device, mc=0, tile=512):
    """x: (C,H,W) standardised. Returns prob (H,W) and std (H,W) (std=0 if mc==0)."""
    model.eval()
    if mc:
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout2d):
                mod.train()
    xt = torch.from_numpy(x[None].astype(np.float32)).to(device)
    H, W = x.shape[1:]
    ph, pw = (-H) % 32, (-W) % 32  # U-Net needs multiples of 32
    xt = torch.nn.functional.pad(xt, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        runs = [torch.sigmoid(model(xt))[0, 0, :H, :W].float().cpu().numpy() for _ in range(max(mc, 1))]
    runs = np.stack(runs)
    return runs.mean(0), runs.std(0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--mode", choices=["fused", "s2", "s1"], default="fused")
    ap.add_argument("--no-robust-train", action="store_true", help="ablation: disable modality/cloud dropout")
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc", type=int, default=20, help="MC-dropout passes at test time")
    ap.add_argument("--out", default="outputs/seg")
    ap.add_argument("--baselines-only", action="store_true")
    a = ap.parse_args()

    if a.download:
        download(a.root); return

    rng = np.random.default_rng(a.seed)
    splits = {k: read_split(a.root, k) for k in SPLITS}
    print({k: len(v) for k, v in splits.items()})
    raw = {k: [load_chip(a.root, c) for c in v] for k, v in splits.items()}

    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)

    # ---------------- index/threshold baselines (no training) ----------------
    base_path = out_root / "baselines.json"
    if not base_path.exists() or a.baselines_only:
        base = {}
        for split in ["test", "bolivia"]:
            meters = {k: SegMeter() for k in ["NDWI>0", "MNDWI>0", "S1 VV<-16dB", "S1 VV Otsu"]}
            for s1, s2, lab in raw[split]:
                meters["NDWI>0"].update(baseline_ndwi(s2), lab)
                meters["MNDWI>0"].update(baseline_mndwi(s2), lab)
                meters["S1 VV<-16dB"].update(baseline_s1_fixed(s1), lab)
                meters["S1 VV Otsu"].update(baseline_s1_otsu(s1), lab)
            base[split] = {k: m.result() for k, m in meters.items()}
        base_path.write_text(json.dumps(base, indent=2))
        print(json.dumps(base, indent=2))
        if a.baselines_only:
            return

    torch, ChipDS, make_model, loss_fn = torch_parts()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s2_use = DEFAULT_S2_USE
    n_s1 = 2 if a.mode == "fused" else 0
    to_xy = lambda lst: [(build_input(s1, s2, a.mode, s2_use), lab) for s1, s2, lab in lst]
    data = {k: to_xy(v) for k, v in raw.items()}
    norm = Normaliser().fit([x for x, _ in data["train"]])
    in_ch = data["train"][0][0].shape[0]
    robust = (a.mode == "fused") and not a.no_robust_train

    run = f"{a.mode}{'' if robust or a.mode != 'fused' else '_norobust'}_{a.encoder}_s{a.seed}"
    out = out_root / run; out.mkdir(parents=True, exist_ok=True)
    (out / "norm.json").write_text(json.dumps({**norm.to_dict(), "mode": a.mode, "s2_use": s2_use,
                                               "encoder": a.encoder}))

    tr = ChipDS(data["train"], norm, a.mode, n_s1, True, a.crop, robust, a.seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
    model = make_model(in_ch, a.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    def evaluate(split, drop_s2=False, cloud=0.0, mc=0, seed=0):
        r = np.random.default_rng(seed)
        meter = SegMeter(); eces = []; err_auc = []
        for x, y in data[split]:
            xx = norm(x).astype(np.float32)
            if a.mode in ("fused", "s2") and (drop_s2 or cloud > 0):
                m = np.ones(y.shape, bool) if drop_s2 else synthetic_cloud_mask(*y.shape, cloud, r)
                xx[n_s1:, m] = 0.0
            prob, std = predict_full(model, xx, torch, device, mc=mc)
            meter.update(prob > 0.5, y)
            if mc:
                eces.append(expected_calibration_error(prob, y))
                v = y != IGNORE
                err = ((prob > 0.5) != (y == 1))[v]
                if 0 < err.mean() < 1:
                    from sklearn.metrics import roc_auc_score
                    err_auc.append(roc_auc_score(err.ravel(), std[v].ravel()))
        res = meter.result()
        if mc:
            res["ECE"] = float(np.mean(eces)); res["UncErrAUROC"] = float(np.nanmean(err_auc)) if err_auc else float("nan")
        return res

    best, log = -1, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        v = evaluate("valid")
        log.append({"epoch": ep, "loss": tot / len(dl), **{f"val_{k}": val for k, val in v.items()}})
        print(f"ep {ep:02d} loss {tot/len(dl):.4f} val IoU {v['IoU']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v["IoU"] > best:
            best = v["IoU"]; torch.save(model.state_dict(), out / "best.pt")
    with open(out / "log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=log[0].keys()); w.writeheader(); w.writerows(log)

    # ---------------- final evaluation with best checkpoint ----------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    metrics = {"run": run, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
               "train_minutes": (time.time() - t0) / 60}
    for split in ["test", "bolivia"]:
        metrics[split] = {"clear": evaluate(split, mc=a.mc)}
        if a.mode == "fused":
            metrics[split]["no_optical"] = evaluate(split, drop_s2=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if a.mode in ("fused", "s2"):  # robustness curve (s2-only model = reference)
        with open(out / "cloud_curve.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["split", "cloud_frac", "IoU", "F1"])
            for split in ["test", "bolivia"]:
                for cf in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]:
                    r = evaluate(split, cloud=cf, seed=123)
                    w.writerow([split, cf, r["IoU"], r["F1"]])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
