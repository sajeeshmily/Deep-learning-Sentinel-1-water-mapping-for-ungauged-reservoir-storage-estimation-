[README.md](https://github.com/user-attachments/files/34062892/README.md)
# Satellite-Driven Reservoir Storage Forecasting with Calibrated Uncertainty — Chennai Reservoir System

Code and results for the methodology: SAR–optical U-Net water segmentation → area-to-storage
conversion → storage forecasting (7/15/30 days) with conformal prediction intervals, evaluated
on Poondi, Cholavaram, Red Hills (Puzhal) and Chembarambakkam.



| Stage | Script | Status |
|---|---|---|
| Storage forecasting, intervals, transfer, satellite-sensitivity (E1–E4) | `src/forecast_experiments.py` | **Run on real data** (Kaggle "Chennai Water Management", 2004–2020). Results in `outputs/forecast/` |
| Earth Engine export (baseline areas, DEM, CHIRPS, ERA5-Land; image stacks) | `gee/01_export_reservoir_data_colab.py` | Phase 1 (tables) **run**; Phase 2 (stacks) ready. The `.js` version is the older Code Editor variant and lacks the fixes below |
| Satellite area vs gauge storage; ungauged rating curves (E-SAT) | `src/satellite_validation.py` | **Run on real Earth Engine exports** (2017–2020). Results in `outputs/satellite/` |
| Forecasting driven by real satellite storage (E5) | `src/satellite_forecast.py` | **Run on real data** (test 2019–2020) |
| U-Net training + segmentation evaluation (Sen1Floods11) | `src/train_unet_sen1floods11.py`, `kaggle_unet_run.ipynb` | **Run on Kaggle** (4 configs x seeds 0,1). Results in `outputs/seg/` (`segmentation_summary.csv`, `F7_cloud_robustness.png`). Weights: `models/`
| Area time series from the trained U-Nets | `src/area_timeseries.py` | Runs S2-only, S1-only and fused models on every reservoir scene; scene logic tested with a stub model. Needs the Phase 2 stacks + weights (run on Kaggle/Colab) |
| Area → storage (gauged rating, regional leave-one-out, DEM hypsometry) | `src/area_to_storage.py` | Code tested on synthetic inputs only; run it on your real satellite areas |

**Important:** E4 uses *simulated* satellite observations (revisit, monsoon cloud loss, SAR
gap-filling, storage error). Report it as a sensitivity analysis, not as satellite results.


## Run order

```bash
# 0. environment (Kaggle/Colab, internet ON)
pip install -q segmentation-models-pytorch rasterio google-cloud-storage scikit-learn scipy pandas matplotlib

# 1. Earth Engine: paste gee/01_export_reservoir_data.js, VERIFY reservoir coordinates,
#    run once with EXPORT_STACKS=false (CSVs), then true (image stacks). Copy Drive files to data/.

# 2. Segmentation (≈1–2 h per run on a T4)
python src/train_unet_sen1floods11.py --download --root sen1floods11
python src/train_unet_sen1floods11.py --root sen1floods11 --baselines-only
for s in 0 1 2; do
  python src/train_unet_sen1floods11.py --root sen1floods11 --mode fused --seed $s
  python src/train_unet_sen1floods11.py --root sen1floods11 --mode fused --no-robust-train --seed $s
  python src/train_unet_sen1floods11.py --root sen1floods11 --mode s2 --seed $s
  python src/train_unet_sen1floods11.py --root sen1floods11 --mode s1 --seed $s
done

# 3. Reservoir area time series
python src/area_timeseries.py --stacks data/stacks --masks data/masks \
    --run outputs/seg/fused_resnet18_s0 --out data/area_timeseries.csv

# 4. Area -> storage (gauge file: date,reservoir,storage in long format)
python src/area_to_storage.py --area data/area_timeseries.csv --gauge data/gauge_long.csv \
    --config configs/reservoirs.json --out outputs/storage --dem-dir data/dem

# 5. Forecasting experiments (CPU, ≈5 min)
python src/forecast_experiments.py --data data --config configs/reservoirs.json --out outputs/forecast
```

## Earth Engine export fixes (made after the first export)

The first Phase 1 export revealed three issues, now fixed in the Colab script; the validation
script handles both the old and new formats:
1. Sentinel-2 cloud mask came from L2A/SCL, whose archive over India starts late 2018, so
   2017–2018 had very few scenes. Now uses s2cloudless (COPERNICUS/S2_CLOUD_PROBABILITY).
2. Chennai sits on two S2 tiles; scenes that only clip a reservoir under-report area. Now a
   per-scene `coverage_frac` is exported and stacks skip scenes with coverage < 95 %.
3. From 2020-03, Sentinel-1 passes are split over two frames. Now mosaicked per day and track.

## Experimental design (forecasting)

- **Splits (chronological):** train 2004–2014 · conformal calibration 2015–2016 · test 2017–2020.
- **Target:** change in storage fraction over h days (h = 7, 15, 30), converted back to mcft.
- **Inputs:** storage lags (0–60 d), 7/30-d changes, own-gauge rainfall sums (1–90 d),
  catchment-mean rainfall, seasonality, headroom. No future rainfall is used.
- **Models:** persistence, day-of-year climatology, AR-ridge, Random Forest, HistGradientBoosting, MLP.
- **Intervals (90 %):** split conformal; Mondrian conformal (season × reservoir); adaptive
  conformal inference with h-day delayed feedback; quantile-regression HistGB baseline.
  All intervals clipped to the physical range [0, 1.05 × capacity].
- **Data QC:** 30 values in 11 isolated error blocks (1–3 days, mostly on the same dates in
  all four reservoirs) were flagged and interpolated — see `outputs/forecast/qc_flagged_spikes.csv`.

## Known limitations 

1. These reservoirs are operated (city supply draw-down, inter-reservoir transfers, Krishna
   water inflow at Poondi). Release/transfer data are not in the public dataset, which caps
   forecast skill, especially at 30 days.
2. No future-rainfall information is used, so monsoon rises are hard to anticipate.
3. Hyperparameters are fixed defaults (not tuned on test data).
4. DEM hypsometry: GLO-30 measures the water surface, so the bed is extrapolated; on a
   synthetic test this underestimated volume by ≈12 %. Treat it as a gauge-free fallback.
5. Capacities in `configs/reservoirs.json` are commonly published values — verify them.

## What is in this package

| Folder | Contents |
|---|---|

| `src/` | All Python scripts (segmentation training, image streaming, rating curves, forecasting, validation, figures) |
| `gee/` | Earth Engine export scripts (Colab and Code Editor versions) |
| `kaggle_unet_run.ipynb`, `colab_stream_inference.ipynb` | Notebooks for U-Net training (Kaggle) and reservoir inference (Colab) |
| `configs/` | Reservoir capacities |
| `data/` | Kaggle Chennai gauge data, Earth Engine exports (`data/gee/`), U-Net reservoir area time series |
| `outputs/forecast/` | Gauge-driven forecasting results (E1–E4), predictions, QC log, figures |
| `outputs/satellite/` | First satellite validation (MNDWI / Otsu from Earth Engine) and optical-only satellite forecasts (E5) |
| `outputs/seg/` | U-Net benchmark metrics (Sen1Floods11), summary table, cloud-robustness figure |
| `outputs/unet_reservoir/` | U-Net reservoir validation and satellite-driven forecasts (E6), figures |
| `outputs/paper/` | Flow chart and block diagram (PNG + PDF) |


