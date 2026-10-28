/**************************************************************************
 * 01_export_reservoir_data.js  —  Google Earth Engine Code Editor script
 *
 * Exports, for each reservoir:
 *   (a) per-scene baseline water area from Sentinel-2 (MNDWI) and Sentinel-1 (VV, Otsu)
 *       + cloud fraction over the reservoir            -> CSV  (baseline_areas_<name>.csv)
 *   (b) per-scene image stacks matching Sen1Floods11 band layout, for U-Net inference
 *       S2: 13 L1C bands (B1..B12, incl. B8A) + SCL    -> GeoTIFF (S2_<name>_<date>.tif)
 *       S1: VV, VH in dB                               -> GeoTIFF (S1_<name>_<date>.tif)
 *   (c) Copernicus GLO-30 DEM over the reservoir       -> GeoTIFF (DEM_<name>.tif)
 *   (d) daily catchment forcing: CHIRPS rainfall, ERA5-Land evaporation & temperature
 *                                                      -> CSV  (forcing_<name>.csv)
 *
 * HOW TO USE
 *   1. Paste into https://code.earthengine.google.com (free non-commercial account).
 *   2. CHECK THE RESERVOIR COORDINATES BELOW. They are approximate centre points.
 *      Better: draw a polygon around each reservoir and replace `geom`.
 *   3. Set EXPORT_STACKS = true only after the CSVs look right — stacks are many files.
 *   4. Run, then click RUN on each task in the Tasks tab. Files land in Google Drive.
 **************************************************************************/

var START = '2017-01-01';
var END   = '2026-09-30';
var DRIVE_FOLDER = 'reservoir_forecast';
var EXPORT_STACKS = false;     // per-scene GeoTIFFs for U-Net inference (large)
var MAX_CLOUD_OVER_RES = 0.6;  // skip S2 stacks with >60% cloud over the reservoir
var BOX_BUFFER_M = 6000;       // search box around the centre point
var CATCHMENT_BUFFER_M = 30000;// crude catchment proxy for forcing; replace with a basin polygon if you have one

// Approximate centre points (lon, lat) — VERIFY on the map before exporting.
var reservoirs = [
  {name: 'POONDI',          lon: 79.860, lat: 13.190},
  {name: 'CHOLAVARAM',      lon: 80.150, lat: 13.220},
  {name: 'REDHILLS',        lon: 80.180, lat: 13.160},   // Puzhal
  {name: 'CHEMBARAMBAKKAM', lon: 80.060, lat: 13.010}
];

var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');

// Reservoir mask = pixels that were EVER water (JRC max extent) inside the box,
// keeping only the connected component touching the centre point.
function reservoirMask(r) {
  var pt = ee.Geometry.Point([r.lon, r.lat]);
  var box = pt.buffer(BOX_BUFFER_M).bounds();
  var maxExt = jrc.select('max_extent').eq(1).clip(box).selfMask();
  var labels = maxExt.connectedComponents(ee.Kernel.plus(1), 1024).select('labels');
  var lbl = labels.reduceRegion({reducer: ee.Reducer.mode(), geometry: pt.buffer(300),
                                 scale: 30, maxPixels: 1e9}).get('labels');
  var mask = labels.eq(ee.Number(lbl)).selfMask();
  // Buffer by 300 m so the mask can capture extents larger than the JRC record.
  var poly = mask.reduceToVectors({geometry: box, scale: 30, maxPixels: 1e10})
                 .geometry().buffer(300);
  return {box: box, geom: poly, pt: pt};
}

// ---------- Sentinel-2 ----------
function s2Collection(geom) {
  var l1c = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
      .filterBounds(geom).filterDate(START, END);
  var sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(geom).filterDate(START, END);
  // Join SR (for SCL cloud mask) to L1C (Sen1Floods11 uses L1C reflectance).
  var join = ee.Join.saveFirst('sr');
  var filt = ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'});
  return ee.ImageCollection(join.apply(l1c, sr, filt));
}

function s2Stats(img, geom) {
  var sr = ee.Image(img.get('sr'));
  var scl = sr.select('SCL');
  var cloud = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  var mndwi = sr.normalizedDifference(['B3', 'B11']);
  var water = mndwi.gt(0).and(cloud.not());
  var px = ee.Image.pixelArea();
  var s = ee.Image.cat(water.multiply(px).rename('water_m2'),
                       cloud.multiply(px).rename('cloud_m2'),
                       px.rename('total_m2'))
      .reduceRegion({reducer: ee.Reducer.sum(), geometry: geom, scale: 20, maxPixels: 1e10});
  return ee.Feature(null, {
    date: img.date().format('YYYY-MM-dd'), sensor: 'S2', id: img.get('system:index'),
    area_km2: ee.Number(s.get('water_m2')).divide(1e6),
    cloud_frac: ee.Number(s.get('cloud_m2')).divide(ee.Number(s.get('total_m2')))
  });
}

// ---------- Sentinel-1 ----------
function s1Collection(geom) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(geom).filterDate(START, END)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .select(['VV', 'VH']);
}

// Otsu threshold on a histogram dictionary (standard GEE recipe).
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means = ee.Array(histogram.get('bucketMeans'));
  var size = means.length().get([0]);
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = sum.divide(total);
  var indices = ee.List.sequence(1, size);
  var bss = indices.map(function (i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans = means.slice(0, 0, i);
    var aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount = total.subtract(aCount);
    var bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
        .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.sort(bss).get([-1]);
}

function s1Stats(img, box, geom) {
  var vv = img.select('VV').focalMedian(30, 'circle', 'meters');  // light speckle filter
  var hist = vv.reduceRegion({reducer: ee.Reducer.histogram(255, 0.1), geometry: box,
                              scale: 20, maxPixels: 1e10}).get('VV');
  var thr = ee.Number(ee.Algorithms.If(hist, otsu(hist), -16));
  // Guard against unimodal scenes: keep threshold within a sensible range.
  thr = thr.max(-22).min(-12);
  var water = vv.lt(thr);
  var a = water.multiply(ee.Image.pixelArea())
      .reduceRegion({reducer: ee.Reducer.sum(), geometry: geom, scale: 20, maxPixels: 1e10});
  return ee.Feature(null, {
    date: img.date().format('YYYY-MM-dd'), sensor: 'S1', id: img.get('system:index'),
    area_km2: ee.Number(a.get('VV')).divide(1e6), cloud_frac: 0, otsu_thr: thr,
    orbit: img.get('orbitProperties_pass')
  });
}

// ---------- Forcing ----------
function forcing(catch_geom) {
  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(START, END);
  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(START, END)
      .select(['total_evaporation_sum', 'temperature_2m', 'potential_evaporation_sum']);
  var rain = chirps.map(function (im) {
    var v = im.reduceRegion({reducer: ee.Reducer.mean(), geometry: catch_geom,
                             scale: 5000, maxPixels: 1e9});
    return ee.Feature(null, {date: im.date().format('YYYY-MM-dd'),
                             rain_mm: v.get('precipitation')});
  });
  var evap = era.map(function (im) {
    var v = im.reduceRegion({reducer: ee.Reducer.mean(), geometry: catch_geom,
                             scale: 11000, maxPixels: 1e9});
    return ee.Feature(null, {date: im.date().format('YYYY-MM-dd'),
      // ERA5 evaporation is in metres of water, negative = evaporation. Convert to mm, positive.
      evap_mm: ee.Number(v.get('total_evaporation_sum')).multiply(-1000),
      pet_mm: ee.Number(v.get('potential_evaporation_sum')).multiply(-1000),
      t2m_c: ee.Number(v.get('temperature_2m')).subtract(273.15)});
  });
  return {rain: rain, evap: evap};
}

// ---------- Main loop ----------
reservoirs.forEach(function (r) {
  var m = reservoirMask(r);
  Map.addLayer(m.geom, {color: 'cyan'}, r.name + ' mask');

  var s2 = s2Collection(m.geom);
  var s1 = s1Collection(m.geom);

  var s2fc = s2.map(function (im) { return s2Stats(im, m.geom); });
  var s1fc = s1.map(function (im) { return s1Stats(im, m.box, m.geom); });
  Export.table.toDrive({collection: s2fc.merge(s1fc), description: 'baseline_areas_' + r.name,
                        folder: DRIVE_FOLDER, fileFormat: 'CSV'});

  var f = forcing(m.pt.buffer(CATCHMENT_BUFFER_M));
  Export.table.toDrive({collection: f.rain, description: 'rain_' + r.name,
                        folder: DRIVE_FOLDER, fileFormat: 'CSV'});
  Export.table.toDrive({collection: f.evap, description: 'evap_' + r.name,
                        folder: DRIVE_FOLDER, fileFormat: 'CSV'});

  var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(m.box)
      .select('DEM').mosaic().clip(m.box);
  Export.image.toDrive({image: dem.toFloat(), description: 'DEM_' + r.name, folder: DRIVE_FOLDER,
                        region: m.box, scale: 30, crs: 'EPSG:32644', maxPixels: 1e10});
  // Also export the reservoir mask polygon (needed by 03 and 04 scripts).
  Export.table.toDrive({collection: ee.FeatureCollection([ee.Feature(m.geom, {name: r.name})]),
                        description: 'mask_' + r.name, folder: DRIVE_FOLDER, fileFormat: 'GeoJSON'});

  if (EXPORT_STACKS) {
    // NOTE: Export.* inside evaluate() is the standard client-side pattern for per-image exports.
    s2fc.filter(ee.Filter.lt('cloud_frac', MAX_CLOUD_OVER_RES)).aggregate_array('id')
      .evaluate(function (ids) {
        ids.forEach(function (id) {
          // 13 L1C bands (Sen1Floods11 layout) + SCL from L2A as band 14 (cloud mask for inference)
          var scl = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + id).select('SCL');
          var im = ee.Image('COPERNICUS/S2_HARMONIZED/' + id)
              .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'])
              .addBands(scl).toUint16();
          Export.image.toDrive({image: im, description: 'S2_' + r.name + '_' + id.slice(0, 15),
            folder: DRIVE_FOLDER + '_stacks', region: m.box, scale: 10, crs: 'EPSG:32644',
            maxPixels: 1e10});
        });
      });
    s1.aggregate_array('system:index').evaluate(function (ids) {
      ids.forEach(function (id) {
        var im = ee.Image('COPERNICUS/S1_GRD/' + id).select(['VV', 'VH']).toFloat();
        Export.image.toDrive({image: im, description: 'S1_' + r.name + '_' + id.slice(17, 32),
          folder: DRIVE_FOLDER + '_stacks', region: m.box, scale: 10, crs: 'EPSG:32644',
          maxPixels: 1e10});
      });
    });
  }
});

Map.setCenter(80.05, 13.12, 10);
