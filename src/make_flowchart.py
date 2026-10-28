"""Methodology flow chart for the paper (PNG 600 dpi + PDF)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"data": "#5b7fa6", "seg": "#2e8b6e", "map": "#c0582b", "hyd": "#7a5aa6", "eval": "#4d4d4d"}
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(7.5, 8.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(29, 140); ax.axis("off")


def box(x, y, w, h, title, lines, kind, fs=6.6, title_fs=7.4):
    col = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=col, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y + h - 4.2), w, 4.2, boxstyle="round,pad=0,rounding_size=1.2", fc=col, ec=col, lw=1.3))
    ax.add_patch(plt.Rectangle((x, y + h - 4.2), w, 2.0, fc=col, ec=col, lw=0))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center", color="white", fontsize=title_fs, fontweight="bold")
    ax.text(x + 1.6, y + h - 5.6, "\n".join(lines), ha="left", va="top", fontsize=fs, linespacing=1.38, color="#222")
    return (x, y, w, h)


def arrow(p, q, col="#555", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def band(y, h, label, col):
    ax.add_patch(plt.Rectangle((0.6, y), 2.6, h, fc=col, ec="none", alpha=.9))
    ax.text(1.9, y + h / 2, label, rotation=90, ha="center", va="center", color="white", fontsize=7, fontweight="bold")


ax.text(51.5, 138.3, "Methodology: gauge-free reservoir storage monitoring and forecasting", ha="center", fontsize=9.2, fontweight="bold")

# ---------------- 1. data  (y 117.5-135.5)
band(117.5, 18, "DATA", C["data"])
d1 = box(4.5, 117.5, 21.5, 18, "Sen1Floods11", ["Hand-labelled water chips", "Sentinel-1 VV/VH +", "Sentinel-2 (13 bands)", "Train / val / test +", "held-out Bolivia"], "data")
d2 = box(27.5, 117.5, 22.5, 18, "Sentinel-1 & Sentinel-2", ["2017–2020, Earth Engine", "S1 GRD (track 92, daily", "  mosaics, ~12-day repeat)", "S2 L1C + s2cloudless", "  cloud probability"], "data")
d3 = box(51.5, 117.5, 21.5, 18, "Reservoir outlines", ["JRC Global Surface", "Water max extent", "Connected component", "+ 300 m buffer", "4 Chennai reservoirs"], "data")
d4 = box(74.5, 117.5, 24.3, 18, "Hydrological data", ["Daily gauge storage &", "  rainfall, 2004–2020", "  (Chennai Water Mgmt.)", "CHIRPS rainfall,", "ERA5-Land evaporation"], "data")

# ---------------- 2. models & preparation  (y 88.5-113.5), channel x 45.5-53.5
band(92.5, 21, "MODELS & PREP.", C["seg"])
s1 = box(4.5, 92.5, 41, 21, "Water segmentation (U-Net, ResNet-18)", [
    "S2-only  |  S1-only  |  fused S1+S2",
    "Fused: with / without robustness training",
    "  (modality dropout + synthetic clouds)",
    "Loss BCE + Dice; MC-dropout uncertainty",
    "Baselines: NDWI, MNDWI, S1 Otsu, −16 dB",
    "Metrics: IoU, F1, ECE, IoU vs cloud (0–100 %)",
    "  on test and held-out Bolivia splits",
    "Free GPU (Kaggle T4), seeds 0 and 1"], "seg", fs=6.4)
h1 = box(53.5, 92.5, 45.3, 21, "Gauge data preparation", [
    "Quality control: 30 isolated spike values",
    "  (1–3-day blocks) removed, interpolated",
    "Storage normalised by reservoir capacity",
    "Chronological splits: train 2004–14 |",
    "  calibration 2015–16 | test 2017–20",
    "Cholavaram gauge flagged (flat record,",
    "  disagrees with satellite): not used for",
    "  fitting, reported separately"], "hyd", fs=6.4)

# ---------------- 3. water mapping  (y 66-84.5), width leaves a right channel for gauge data
band(73.5, 15, "WATER MAPPING", C["map"])
m1 = box(4.5, 73.5, 83, 15, "Reservoir water area time series (images streamed from Earth Engine, no export)", [
    "Every scene on a 10 m UTM grid; six methods on identical pixels:",
    "  S2 U-Net, S1 U-Net, fused U-Net, MNDWI, NDWI, S1 Otsu",
    "Cloud fraction from s2cloudless (> 50 %); optical methods kept only if < 10 % cloud over the reservoir",
    "Sensor switching: S2 U-Net on clear dates, S1 U-Net on all other dates (cloud-independent, ~12-day revisit)",
    "Output: water area (km²) per reservoir, date and method, with MC-dropout spread"], "map", fs=6.4)

# ---------------- 4. storage & forecasts  (y 35.5-62)
band(46.5, 23, "STORAGE & FORECASTS", C["hyd"])
a1 = box(4.5, 46.5, 40, 23, "Area → storage (rating curves)", [
    "Power law  S = a + b·A^c",
    "Gauged: own gauge record,",
    "  time-blocked cross-validation",
    "Ungauged (gauge-free): curve fitted",
    "  on OTHER reservoirs, S/capacity",
    "  vs A/max area (leave-one-out)",
    "Dry-bed false-water check",
    "  (area when storage < 5 % capacity)"], "hyd", fs=6.4)
f1 = box(47.5, 46.5, 51.3, 23, "Storage forecasting (7, 15, 30 days)", [
    "Inputs: storage lags (0–60 d), changes, rainfall",
    "  sums (1–90 d), seasonality, days since last obs.",
    "Models: persistence, climatology, AR-ridge,",
    "  random forest, HistGB, MLP",
    "Satellite-like training inputs (revisit, gaps, noise)",
    "90 % intervals: split, Mondrian (season × reservoir),",
    "  adaptive conformal; quantile regression",
    "Gauge-free: forecaster trained on other reservoirs",
    "  + CHIRPS rainfall + satellite storage"], "hyd", fs=6.4)

# ---------------- 5. evaluation  (y 20-32)
band(30.5, 12, "EVALUATION", C["eval"])
e1 = box(4.5, 30.5, 94.3, 12, "Evaluation against gauge storage", [
    "Storage and forecasts: NSE, KGE, RMSE, MAE, PBIAS; Diebold–Mariano test vs persistence; by season (dry, SW, NE monsoon)",
    "Intervals: coverage (PICP), width (MPIW), Winkler score; transfer to unseen reservoirs (leave-one-reservoir-out)",
    "Error budget: daily gauge  vs  satellite dates with exact storage  vs  satellite-derived storage"], "eval", fs=6.4)

# ---------------- arrows
def lab(x, y, t, col, ha="left"):
    ax.text(x, y, t, fontsize=5.8, color=col, va="center", ha=ha, style="italic")

arrow((15.25, 117.5), (15.25, 113.5), C["seg"])
arrow((86.65, 117.5), (86.65, 113.5), C["hyd"])
# imagery + outlines merge into the central channel
ax.plot([38.75, 38.75, 49.5], [117.5, 115.5, 115.5], color=C["map"], lw=1.2)
ax.plot([62.25, 62.25, 49.5], [117.5, 115.5, 115.5], color=C["map"], lw=1.2)
arrow((49.5, 115.5), (49.5, 88.5), C["map"])
ax.text(49.5, 103, "imagery\n+\noutlines", fontsize=5.8, color=C["map"], ha="center", va="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=0.4))
arrow((25, 92.5), (25, 88.5), C["seg"]); lab(26, 90.5, "trained models", C["seg"])
arrow((24.5, 73.5), (24.5, 69.5), C["map"]); lab(25.5, 71.5, "water area", C["map"])
# gauge storage down the right channel to both hydrology boxes
ax.plot([93.15, 93.15], [92.5, 71.5], color=C["hyd"], lw=1.2)
arrow((93.15, 71.5), (93.15, 69.5), C["hyd"])
ax.text(93.15, 81, "gauge\nstorage &\nrainfall", fontsize=5.6, color=C["hyd"], ha="center", va="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=0.3))
ax.plot([93.15, 46], [71.5, 71.5], color=C["hyd"], lw=1.0, ls=(0, (2, 1.5)))
arrow((46, 71.5), (40, 69.5), C["hyd"], lw=1.0)
arrow((44.5, 58), (47.5, 58), C["hyd"]); lab(46, 60.3, "storage", C["hyd"], ha="center")
arrow((24.5, 46.5), (24.5, 42.5), C["eval"]); arrow((73.15, 46.5), (73.15, 42.5), C["eval"])

for ext in ("png", "pdf"):
    fig.savefig(f"/home/claude/reservoir_forecast/outputs/paper/Fig1_methodology_flowchart.{ext}", dpi=600 if ext == "png" else None)
print("ok")
