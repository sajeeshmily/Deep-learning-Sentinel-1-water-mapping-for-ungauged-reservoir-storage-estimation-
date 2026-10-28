"""Simple block diagram of the method (PNG 600 dpi + PDF)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MAIN, SIDE, OUT = "#2f5f8a", "#e9eef4", "#c0582b"
fig = plt.figure(figsize=(7.5, 3.0)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")


def block(x, y, w, h, text, fc, tc="white", fs=7.6, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc,
                                ec=MAIN if fc == SIDE else fc, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(p, q, col="#444"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3, shrinkA=0, shrinkB=0))


# main chain
W, H, Y, GAP = 16.2, 11, 13, 3.3
xs = [2.15 + i * (W + GAP) for i in range(5)]
labels = ["Sentinel-1 &\nSentinel-2\nimagery",
          "U-Net water\nsegmentation\n(sensor switching)",
          "Water area\ntime series",
          "Area → storage\n(rating curve)",
          "Storage\nforecast\n(7–30 days)"]
for i, (x, t) in enumerate(zip(xs, labels)):
    block(x, Y, W, H, t, OUT if i == 4 else MAIN)
    if i:
        arrow((xs[i - 1] + W, Y + H / 2), (x, Y + H / 2))

# side inputs (top)
block(xs[1], 30, W, 7.5, "Sen1Floods11\n(training data)", SIDE, tc="#1d2b3a", fs=7, bold=False)
arrow((xs[1] + W / 2, 30), (xs[1] + W / 2, Y + H))
block(xs[3] + 3, 30, xs[4] + W - xs[3] - 6, 7.5, "Gauge storage & rainfall\n(other reservoirs)", SIDE, tc="#1d2b3a", fs=7, bold=False)
arrow((xs[3] + W / 2, 30), (xs[3] + W / 2, Y + H))
arrow((xs[4] + W / 2, 30), (xs[4] + W / 2, Y + H))

# evaluation (bottom)
block(xs[2], 1.5, xs[4] + W - xs[2], 6.5, "Evaluation against gauge storage  (NSE, KGE, RMSE, interval coverage)",
      SIDE, tc="#1d2b3a", fs=7, bold=False)
for x in (xs[3], xs[4]):
    arrow((x + W / 2, Y), (x + W / 2, 8.0))

for ext in ("png", "pdf"):
    fig.savefig(f"/home/claude/reservoir_forecast/outputs/paper/Fig_block_diagram.{ext}", dpi=600 if ext == "png" else None)
print("ok")
