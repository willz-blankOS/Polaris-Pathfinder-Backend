# co2_exposure.py
# Build a CO2 exposure index from OSM roads + HotGrid winds, then score a route.

import math
import zarr
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin, rowcol
from rasterio.features import rasterize
from shapely.geometry import LineString
from pyproj import Transformer, CRS
import osmnx as ox
import geopandas as gpd
from scipy.ndimage import gaussian_filter, rotate

# ---------------- utils ----------------

def utm_epsg_for_lonlat(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return int(f"{326 if lat >= 0 else 327}{zone:02d}")

def meters_per_degree_lon(lat_deg):
    return 111_320.0 * math.cos(math.radians(lat_deg))

def _ensure_edge_geometry(G):
    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" not in d:
            p = (G.nodes[u]["x"], G.nodes[u]["y"])
            q = (G.nodes[v]["x"], G.nodes[v]["y"])
            d["geometry"] = LineString([p, q])

def _parse_int(x, default):
    try:
        if isinstance(x, (list, tuple)):
            x = x[0]
        return int(str(x).split()[0])
    except Exception:
        return default

# ---------------- step 1: emissions raster ----------------

def _road_emissions_gdf(bbox_ll, target_crs):
    """Download OSM roads for bbox (lon/lat), build per-geometry emission attributes, project to target_crs."""
    west, south, east, north = bbox_ll
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west,
                           network_type="drive", simplify=True)  # vehicle roads
    _ensure_edge_geometry(G)
    edges = ox.graph_to_gdfs(G, nodes=False)
    if edges.empty:
        return gpd.GeoDataFrame({"emission": []}, geometry=[], crs="EPSG:4326").to_crs(target_crs)

    # Class weights (heuristic; scale roughly by expected AADT)
    base_w = {
        "motorway": 1.00, "trunk": 0.85, "primary": 0.65, "secondary": 0.45,
        "tertiary": 0.30, "unclassified": 0.20, "residential": 0.15,
        "service": 0.10, "living_street": 0.08
    }

    # Prepare attributes
    hw = edges.get("highway")
    if isinstance(hw, (list, tuple)) or edges["highway"].dtype == object:
        # OSM can store multiple highway tags; pick first if list
        edges["hw_simple"] = edges["highway"].apply(lambda h: h[0] if isinstance(h, list) else h)
    else:
        edges["hw_simple"] = edges["highway"]

    edges["lanes_i"] = edges.get("lanes", 2).apply(lambda v: _parse_int(v, 2))
    edges["maxspeed_i"] = edges.get("maxspeed", 50).apply(lambda v: _parse_int(v, 50))

    def class_weight(h):
        return base_w.get(h, 0.15)

    # Emission per *square meter* proxy (unitless index):
    #   e_area = base(class) * lanes * f(speed)
    # with f(speed) ~ 1 at 50 km/h, larger above.
    edges["e_area"] = (
        edges["hw_simple"].apply(class_weight)
        * np.clip(edges["lanes_i"], 1, 8)
        * np.clip(edges["maxspeed_i"] / 50.0, 0.6, 2.0)
    )

    # Project to target CRS (meters), then buffer to approximate paved width
    gdf = edges[["geometry", "e_area"]].copy()
    gdf = gdf.set_crs(4326, allow_override=True).to_crs(target_crs)

    # Width ≈ 3.5 m per lane + shoulder; cap 4–30 m
    widths = np.clip(edges["lanes_i"] * 3.5 + 2.0, 4.0, 30.0).values
    gdf["geometry"] = gdf["geometry"].buffer(widths/2.0, cap_style=2)  # m → half-width
    gdf = gdf[gdf.geometry.notnull() & ~gdf.is_empty]
    gdf = gdf.rename(columns={"e_area": "emission"})
    return gdf

def _rasterize_emissions(gdf, out_crs, out_transform, out_h, out_w):
    """Sum emission polygons into a raster (float32)."""
    shapes = ((geom, float(val)) for geom, val in zip(gdf.geometry.values, gdf["emission"].values))
    arr = rasterize(
        shapes=shapes,
        out_shape=(out_h, out_w),
        fill=0.0,
        transform=out_transform,
        all_touched=True,
        dtype="float32",
        merge_alg=rio.enums.MergeAlg.add
    )
    # Normalize to 0..1 using 99th percentile to keep hot spots but avoid outliers
    q = np.nanpercentile(arr[arr > 0], 99) if np.any(arr > 0) else 1.0
    arr = arr / (q + 1e-6)
    return np.clip(arr, 0.0, 1.0)

# ---------------- step 2: wind-aware dispersion ----------------

def _anisotropic_dispersion(emissions, pixel_m, wind_u, wind_v,
                            base_sigma_m=75.0, k_par=50.0, k_perp=15.0):
    """
    Simple line-source dispersion proxy:
      - Rotate so wind points +X
      - Blur with σ_parallel (along wind) and σ_perp (across wind)
      - Rotate back
    """
    H, W = emissions.shape
    speed = math.hypot(wind_u, wind_v)
    if speed < 0.2:
        # calm → isotropic blur
        sigma = max(1.0, base_sigma_m / pixel_m)
        return gaussian_filter(emissions, sigma=sigma, mode="nearest")

    theta = math.degrees(math.atan2(wind_v, wind_u))  # radians → deg
    # Rotate grid so wind aligns with +X (angle negative because ndimage.rotate rotates CCW)
    rot = rotate(emissions, angle=-theta, reshape=True, order=1, mode="nearest", prefilter=False)
    # Pixel sigmas
    sigma_par  = max(1.0, (base_sigma_m + k_par * speed)  / pixel_m)
    sigma_perp = max(1.0, (base_sigma_m + k_perp * speed) / pixel_m)
    # Apply separable blur (y,x) because after rotation x aligns with wind
    blurred = gaussian_filter(rot, sigma=(sigma_perp, sigma_par), mode="nearest")
    # Rotate back to original grid size
    back = rotate(blurred, angle=+theta, reshape=True, order=1, mode="nearest", prefilter=False)
    # Center-crop to original HxW
    y0 = (back.shape[0] - H) // 2
    x0 = (back.shape[1] - W) // 2
    back = back[y0:y0+H, x0:x0+W]
    # Normalize again to 0..1 for a clean index
    m = back.max()
    return back / (m + 1e-6)

# ---------------- HotGrid wind slice ----------------

def _mean_wind_from_hotgrid(hotgrid_path, bbox_ll):
    """Read latest slot wind_u, wind_v from your Zarr HotGrid and average over bbox."""
    z = zarr.open(hotgrid_path, mode="r")
    attrs = z.attrs
    west, south = attrs["bounds"]["west"], attrs["bounds"]["south"]
    east, north = attrs["bounds"]["east"], attrs["bounds"]["north"]
    d = float(attrs["delta_deg"])
    slot = attrs["time_index"][attrs["latest_ts"]]
    bands = attrs["bands"]; bu = bands.index("wind_u"); bv = bands.index("wind_v")

    # Map bbox (lon/lat) to row/col
    def rc(lon, lat):
        c = int((lon - west)/d); r = int((lat - south)/d)
        r = max(0, min(z["grid"].shape[2]-1, r))
        c = max(0, min(z["grid"].shape[3]-1, c))
        return r, c

    (w,s,e,n) = bbox_ll
    r0,c0 = rc(w,s); r1,c1 = rc(e,n)
    rmin, rmax = min(r0,r1), max(r0,r1)
    cmin, cmax = min(c0,c1), max(c0,c1)

    u = z["grid"][slot, bu, rmin:rmax+1, cmin:cmax+1].astype("float32")
    v = z["grid"][slot, bv, rmin:rmax+1, cmin:cmax+1].astype("float32")
    return float(np.nanmean(u)), float(np.nanmean(v))

# ---------------- public API ----------------

def build_co2_field(bbox_ll, res_m=100.0, hotgrid_path="hotgrid_gta.zarr"):
    """
    Build a CO2 exposure index raster (0..1) for bbox (lon/lat) at res_m.
    Steps: OSM roads → emission → wind from HotGrid → anisotropic dispersion.
    Returns: (co2_idx, transform, crs)
    """
    # 1) Choose UTM CRS for bbox center
    lonc = 0.5*(bbox_ll[0] + bbox_ll[2]); latc = 0.5*(bbox_ll[1] + bbox_ll[3])
    epsg = utm_epsg_for_lonlat(lonc, latc)
    crs = CRS.from_epsg(epsg)

    # 2) Project bbox to meters; set raster grid
    T_ll_to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    Wm, Sm = T_ll_to_utm.transform(bbox_ll[0], bbox_ll[1])
    Em, Nm = T_ll_to_utm.transform(bbox_ll[2], bbox_ll[3])
    width = max(1, int(math.ceil((Em - Wm) / res_m)))
    height = max(1, int(math.ceil((Nm - Sm) / res_m)))
    transform = from_origin(Wm, Nm, res_m, res_m)

    # 3) OSM → emissions (projected)
    gdf = _road_emissions_gdf(bbox_ll, crs)
    emissions = _rasterize_emissions(gdf, crs, transform, height, width)  # 0..1

    # 4) Winds (mean over bbox) from HotGrid
    wind_u, wind_v = _mean_wind_from_hotgrid(hotgrid_path, bbox_ll)

    # 5) Dispersion → CO2 exposure index (0..1)
    co2_idx = _anisotropic_dispersion(emissions, pixel_m=res_m, wind_u=wind_u, wind_v=wind_v)

    return co2_idx.astype("float32"), transform, crs

def route_co2_exposure(co2_idx, transform, route_lonlat, crs):
    """
    Length-weighted mean CO2 index along a route (lon/lat list).
    Returns float in [0,1].
    """
    T_ll_to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xy = [T_ll_to_utm.transform(lon, lat) for lon, lat in route_lonlat]
    ls = LineString(xy)
    if ls.length <= 0:
        return 0.0
    # densify ~25 m
    n = max(2, int(math.ceil(ls.length / 25.0)) + 1)
    dists = np.linspace(0.0, ls.length, n)
    pts = [ls.interpolate(d).coords[0] for d in dists]

    # length weights
    segs = [math.hypot(x2-x1, y2-y1) for (x1,y1),(x2,y2) in zip(pts[:-1], pts[1:])]
    wts = np.array(segs + [0.0], dtype=np.float32)[:, None]

    # sample raster
    H, W = co2_idx.shape
    rc = [rowcol(transform, x, y) for x, y in pts]
    rc = [(max(0, min(H-1, r)), max(0, min(W-1, c))) for r, c in rc]
    vals = np.array([co2_idx[r, c] for r, c in rc], dtype=np.float32)[:, None]

    mask = np.isfinite(vals).astype(np.float32)
    num = np.nansum(vals * wts * mask, axis=0)
    den = np.nansum(wts * mask, axis=0) + 1e-9
    return float(num/den)
