"""Shared pieces for water segmentation: band config, normalisation, metrics, baselines.

Pure numpy so it can be imported (and unit-tested) without PyTorch.
"""
import numpy as np

# Sen1Floods11 S2Hand band order (13 Sentinel-2 L1C bands)
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

# Lightweight default: 6 S2 bands most informative for water + both S1 bands.
DEFAULT_S2_USE = ["B2", "B3", "B4", "B8", "B11", "B12"]

IGNORE = -1  # Sen1Floods11 "no data" label


def s2_idx(names):
    return [S2_BANDS.index(n) for n in names]


def prep_s1(s1):
    """S1 in dB (VV, VH). Clip extreme values and replace NaN."""
    s1 = np.nan_to_num(s1.astype(np.float32), nan=-30.0)
    return np.clip(s1, -50.0, 5.0)


def prep_s2(s2, use=DEFAULT_S2_USE):
    """S2 L1C digital numbers (reflectance x 10000) -> reflectance, chosen bands."""
    s2 = np.nan_to_num(s2.astype(np.float32), nan=0.0)[s2_idx(use)]
    if np.nanpercentile(s2, 99) > 2.0:   # digital numbers (x10000) -> reflectance; already 0-1 otherwise
        s2 = s2 / 10000.0
    return np.clip(s2, 0.0, 1.5)


# ----------------------------------------------------------------------------
# Metrics. Sen1Floods11 convention: pool pixels over the whole split
# ("total IoU"), ignoring label == -1. We also report per-chip mean IoU.
# ----------------------------------------------------------------------------
class SegMeter:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.chip_iou = []

    def update(self, pred, label):
        valid = label != IGNORE
        p = pred.astype(bool) & valid
        t = (label == 1) & valid
        tp = int((p & t).sum()); fp = int((p & ~t).sum())
        fn = int((~p & t & valid).sum()); tn = int((~p & ~t & valid).sum())
        self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn
        if tp + fp + fn > 0:
            self.chip_iou.append(tp / (tp + fp + fn))

    def result(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        return {"IoU": iou, "F1": f1, "Precision": prec, "Recall": rec,
                "mIoU_chip": float(np.mean(self.chip_iou)) if self.chip_iou else float("nan"),
                "PixelAcc": acc}


def otsu_threshold(x, bins=256):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    w0 = np.cumsum(w); w1 = w0[-1] - w0
    m0 = np.cumsum(w * centers) / np.maximum(w0, 1)
    m1 = (np.sum(w * centers) - np.cumsum(w * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return centers[np.argmax(between)]


# ----------------------------------------------------------------------------
# Index / threshold baselines (no training). Inputs are RAW (not standardised).
# ----------------------------------------------------------------------------
def baseline_ndwi(s2_raw):
    g, n = s2_raw[S2_BANDS.index("B3")].astype(np.float32), s2_raw[S2_BANDS.index("B8")].astype(np.float32)
    return ((g - n) / np.maximum(g + n, 1e-6)) > 0


def baseline_mndwi(s2_raw):
    g, s = s2_raw[S2_BANDS.index("B3")].astype(np.float32), s2_raw[S2_BANDS.index("B11")].astype(np.float32)
    return ((g - s) / np.maximum(g + s, 1e-6)) > 0


def baseline_s1_fixed(s1_raw, thr=-16.0):
    return prep_s1(s1_raw)[0] < thr


def baseline_s1_otsu(s1_raw):
    vv = prep_s1(s1_raw)[0]
    t = otsu_threshold(vv)
    t = np.clip(t, -22, -12) if np.isfinite(t) else -16.0  # guard unimodal chips
    return vv < t


# ----------------------------------------------------------------------------
# Synthetic cloud masks for the "accuracy vs cloud cover" robustness curve.
# Smooth random fields thresholded to a target cover fraction.
# ----------------------------------------------------------------------------
def synthetic_cloud_mask(h, w, frac, rng):
    if frac <= 0:
        return np.zeros((h, w), bool)
    if frac >= 1:
        return np.ones((h, w), bool)
    coarse = rng.random((max(h // 32, 2), max(w // 32, 2)))
    # bilinear upsample via np.kron + simple box blur (keeps numpy-only)
    up = np.kron(coarse, np.ones((32, 32)))[:h, :w]
    k = 15
    pad = np.pad(up, k, mode="reflect")
    cs = pad.cumsum(0).cumsum(1)
    blur = (cs[2*k:, 2*k:] - cs[:-2*k, 2*k:] - cs[2*k:, :-2*k] + cs[:-2*k, :-2*k]) / (2*k) ** 2
    blur = blur[:h, :w]
    thr = np.quantile(blur, 1 - frac)
    return blur >= thr


def expected_calibration_error(prob, label, n_bins=15):
    valid = label != IGNORE
    p = prob[valid].ravel(); y = (label[valid] == 1).ravel().astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
