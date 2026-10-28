"""
01_export_reservoir_data_colab.py — Earth Engine export, Python API version for Google Colab.

Same outputs as 01_export_reservoir_data.js, but every export task is STARTED AUTOMATICALLY
(the Code Editor needs a manual "Run" click per task, impractical for ~1,000 image exports).

Setup (once):
  1. Register for Earth Engine (free, non-commercial): https://code.earthengine.google.com/register
     This creates/links a Google Cloud project; note its project ID.
  2. In Colab:   !pip install -q earthengine-api
                 then run this file's cells (or: %run 01_export_reservoir_data_colab.py)
  3. Watch progress: https://code.earthengine.google.com/tasks  (or ee.data.listOperations())

Two phases (set PHASE below):
  PHASE = "tables"  -> small CSVs + DEM + masks (minutes). Upload these to Claude first:
                       baseline_areas_<RES>.csv, rain_<RES>.csv, evap_<RES>.csv
  PHASE = "stacks"  -> per-scene GeoTIFFs for the U-Net (hours; a few GB in Drive)
"""
import ee
from collections import Counter, defaultdict
from datetime import datetime, timezone

PROJECT = "your-gcp-project-id"   # <-- change
PHASE = "tables"                  # "tables" or "stacks"
DRY_RUN = False                   # True: only count the exports, start nothing

# Gauge data (Kaggle Chennai) ends 2020-12; 2017-2020 gives the validation overlap.
# Extend END to the present if you only need the forecasting time series.
START, END = "2017-01-01", "2020-12-31"
DRIVE_FOLDER = "reservoir_forecast"
MAX_CLOUD_OVER_RES = 0.6
S2_EXPORT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # = DEFAULT_S2_USE in src/water_seg_common.py
MAX_TILE_CLOUD = 70        # stacks: skip S2 tiles that are >70 % cloudy (metadata; whole 110 km tile)
BOX_BUFFER_M = 6000
CATCHMENT_BUFFER_M = 30000
CRS = "EPSG:32644"

# Approximate centre points (lon, lat) — VERIFY on a map; replace with drawn polygons if possible.
RESERVOIRS = {
    "POONDI":          (79.860, 13.190),
    "CHOLAVARAM":      (80.150, 13.220),
    "REDHILLS":        (80.180, 13.160),   # Puzhal
    "CHEMBARAMBAKKAM": (80.060, 13.010),
}

try:
    ee.Initialize(project=PROJECT)
except Exception:
    ee.Authenticate()          # opens a Google sign-in link in Colab
    ee.Initialize(project=PROJECT)

JRC = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")


def reservoir_mask(lon, lat):
    pt = ee.Geometry.Point([lon, lat])
    box = pt.buffer(BOX_BUFFER_M).bounds()
    max_ext = JRC.select("max_extent").eq(1).clip(box).selfMask()
    labels = max_ext.connectedComponents(ee.Kernel.plus(1), 1024).select("labels")
    lbl = labels.reduceRegion(reducer=ee.Reducer.mode(), geometry=pt.buffer(300),
                              scale=30, maxPixels=1e9).get("labels")
    mask = labels.eq(ee.Number(lbl)).selfMask()
    poly = mask.reduceToVectors(geometry=box, scale=30, maxPixels=1e10).geometry().buffer(300)
    return pt, box, poly


CLOUD_PROB_THR = 50  # s2cloudless probability (%) above which a pixel is cloud


def s2_collection(geom):
    """Sentinel-2 L1C joined with s2cloudless cloud probability. Both cover the full archive
    from 2017 (unlike L2A/SCL, which over India only starts in late 2018)."""
    l1c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(START, END)
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(geom).filterDate(START, END)
    join = ee.Join.saveFirst("cloud_prob")
    filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
    return ee.ImageCollection(join.apply(l1c, prob, filt))


def s2_stats(geom):
    def f(img):
        cp = ee.Image(img.get("cloud_prob")).select("probability")
        cloud = cp.gt(CLOUD_PROB_THR)
        covered = img.select("B3").mask().gt(0)            # inside this tile's footprint
        water = img.normalizedDifference(["B3", "B11"]).gt(0).And(cloud.Not()).And(covered)
        px = ee.Image.pixelArea()
        s = ee.Image.cat(water.multiply(px).rename("water_m2"),
                         cloud.And(covered).multiply(px).rename("cloud_m2"),
                         covered.multiply(px).rename("covered_m2"),
                         px.rename("total_m2")).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=20, maxPixels=1e10)
        total = ee.Number(s.get("total_m2"))
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"), "sensor": "S2", "id": img.get("system:index"),
            "area_km2": ee.Number(s.get("water_m2")).divide(1e6),
            "cloud_frac": ee.Number(s.get("cloud_m2")).divide(total),
            "coverage_frac": ee.Number(s.get("covered_m2")).divide(total)})
    return f


def s1_collection(geom):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(START, END)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))


def s1_daily(geom):
    """One mosaic per day and relative orbit, so a pass split across two frames is measured once."""
    col = s1_collection(geom)
    keys = col.map(lambda im: im.set("key", ee.String(im.date().format("YYYYMMdd")).cat("_")
                                     .cat(ee.Number(im.get("relativeOrbitNumber_start")).format("%d"))))
    def mk(k):
        k = ee.String(k)
        sub = keys.filter(ee.Filter.eq("key", k))
        first = ee.Image(sub.first())
        return (sub.mosaic().setDefaultProjection(first.projection())
                .set({"system:time_start": first.get("system:time_start"), "id": k,
                      "track": first.get("relativeOrbitNumber_start"), "n_frames": sub.size()}))
    return ee.ImageCollection.fromImages(keys.aggregate_array("key").distinct().map(mk))


def otsu(histogram):
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get("histogram"))
    means = ee.Array(histogram.get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    total_sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = total_sum.divide(total)

    def bss(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_mean = means.slice(0, 0, i).multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        b_count = total.subtract(a_count)
        b_mean = total_sum.subtract(a_count.multiply(a_mean)).divide(b_count)
        return a_count.multiply(a_mean.subtract(mean).pow(2)).add(b_count.multiply(b_mean.subtract(mean).pow(2)))

    scores = ee.List.sequence(1, size).map(bss)
    return means.sort(ee.Array(scores)).get([-1])


def s1_stats(box, geom):
    def f(img):
        vv = img.select("VV").focalMedian(30, "circle", "meters")
        hist = vv.reduceRegion(reducer=ee.Reducer.histogram(255, 0.1), geometry=box,
                               scale=20, maxPixels=1e10).get("VV")
        thr = ee.Number(ee.Algorithms.If(hist, otsu(hist), -16)).max(-22).min(-12)
        a = vv.lt(thr).multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=20, maxPixels=1e10)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"), "sensor": "S1", "id": img.get("id"),
            "area_km2": ee.Number(a.get("VV")).divide(1e6), "cloud_frac": 0, "otsu_thr": thr,
            "track": img.get("track"), "n_frames": img.get("n_frames")})
    return f


def forcing(catch):
    def rain(im):
        v = im.reduceRegion(reducer=ee.Reducer.mean(), geometry=catch, scale=5000, maxPixels=1e9)
        return ee.Feature(None, {"date": im.date().format("YYYY-MM-dd"), "rain_mm": v.get("precipitation")})

    def evap(im):
        v = im.reduceRegion(reducer=ee.Reducer.mean(), geometry=catch, scale=11000, maxPixels=1e9)
        return ee.Feature(None, {"date": im.date().format("YYYY-MM-dd"),
                                 "evap_mm": ee.Number(v.get("total_evaporation_sum")).multiply(-1000),
                                 "pet_mm": ee.Number(v.get("potential_evaporation_sum")).multiply(-1000),
                                 "t2m_c": ee.Number(v.get("temperature_2m")).subtract(273.15)})
    ch = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(START, END).map(rain)
    er = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(START, END)
          .select(["total_evaporation_sum", "temperature_2m", "potential_evaporation_sum"]).map(evap))
    return ch, er


def start(task):
    if not DRY_RUN:
        task.start()
    return task


tasks = []
for name, (lon, lat) in RESERVOIRS.items():
    pt, box, poly = reservoir_mask(lon, lat)
    region = poly.bounds()  # export only the reservoir footprint, not the whole search box

    if PHASE == "tables":
        s2 = s2_collection(poly)
        fc = s2.map(s2_stats(poly)).merge(s1_daily(poly).map(s1_stats(box, poly)))
        tasks.append(start(ee.batch.Export.table.toDrive(
            collection=fc, description=f"baseline_areas_{name}", folder=DRIVE_FOLDER, fileFormat="CSV")))
        rain, evap = forcing(pt.buffer(CATCHMENT_BUFFER_M))
        tasks.append(start(ee.batch.Export.table.toDrive(collection=rain, description=f"rain_{name}",
                                                         folder=DRIVE_FOLDER, fileFormat="CSV")))
        tasks.append(start(ee.batch.Export.table.toDrive(collection=evap, description=f"evap_{name}",
                                                         folder=DRIVE_FOLDER, fileFormat="CSV")))
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(box).select("DEM").mosaic().clip(region)
        tasks.append(start(ee.batch.Export.image.toDrive(image=dem.toFloat(), description=f"DEM_{name}",
                                                         folder=DRIVE_FOLDER, region=region, scale=30,
                                                         crs=CRS, maxPixels=1e10)))
        tasks.append(start(ee.batch.Export.table.toDrive(
            collection=ee.FeatureCollection([ee.Feature(poly, {"name": name})]),
            description=f"mask_{name}", folder=DRIVE_FOLDER, fileFormat="GeoJSON")))

    elif PHASE == "stacks":
        # Scene selection uses METADATA ONLY (no per-pixel statistics), which stays within the
        # interactive memory limit. Cloud over the reservoir is measured later, per scene, from
        # band 14 (s2cloudless probability) during U-Net inference.
        reg_info = region.getInfo()                       # small rectangle, computed once
        region_c = ee.Geometry(reg_info)
        s2 = (s2_collection(region_c)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_TILE_CLOUD)))  # join keeps only scenes with a cloud-probability image
        # keep only tiles whose footprint contains the whole export rectangle
        s2 = s2.map(lambda im: im.set("covers", im.geometry().contains(region_c, ee.ErrorMargin(50))))
        s2 = s2.filter(ee.Filter.eq("covers", True))
        meta2 = s2.reduceColumns(ee.Reducer.toList(2), ["system:index", "CLOUDY_PIXEL_PERCENTAGE"]).get("list").getInfo()
        best = {}                                     # one tile per date (least cloudy) -> no duplicates
        for sid, cc in meta2:
            if sid[:8] not in best or cc < best[sid[:8]][1]:
                best[sid[:8]] = (sid, cc)
        sids = sorted(v[0] for v in best.values())
        for sid in sids:
            cp = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{sid}").select("probability")
            # COMPACT layout (7 bands): the 6 bands the U-Nets use + s2cloudless probability (0-100)
            im = (ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}")
                  .select(S2_EXPORT_BANDS).addBands(cp).toUint16())
            tasks.append(start(ee.batch.Export.image.toDrive(
                image=im, description=f"S2_{name}_{sid[:15]}_{sid[-6:]}", folder=DRIVE_FOLDER + "_stacks",
                region=region_c, scale=10, crs=CRS, maxPixels=1e10)))
        # Sentinel-1: list scene metadata, group frames per day and track client-side, mosaic each group
        s1c = s1_collection(region_c)
        meta = s1c.reduceColumns(ee.Reducer.toList(3),
                                 ["system:index", "system:time_start", "relativeOrbitNumber_start"]).get("list").getInfo()
        groups = defaultdict(list)
        for sid1, t_ms, trk in meta:
            day = datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime("%Y%m%d")
            groups[(day, int(trk))].append(sid1)
        main_track = Counter(trk for _, trk in groups).most_common(1)[0][0]
        n_s1 = 0
        for (day, trk), ids in sorted(groups.items()):
            if trk != main_track:
                continue
            # backscatter in centi-dB as int16 (0.01 dB precision; half the size of float32); no-data = -9999
            im = (ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{x}") for x in ids]).mosaic()
                  .select(["VV", "VH"]).multiply(100).round().unmask(-9999).toInt16())
            tasks.append(start(ee.batch.Export.image.toDrive(
                image=im, description=f"S1_{name}_{day}T000000", folder=DRIVE_FOLDER + "_stacks",
                region=region_c, scale=10, crs=CRS, maxPixels=1e10)))
            n_s1 += 1
        print(f"  {name}: {len(sids)} S2 scenes, {n_s1} S1 dates (track {main_track})")
    print(name, "queued; total tasks so far:", len(tasks))

print(f"{len(tasks)} export tasks {'counted (DRY RUN, nothing started)' if DRY_RUN else 'started'} -> Google Drive/{DRIVE_FOLDER}{'_stacks' if PHASE == 'stacks' else ''}")
