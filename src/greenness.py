import math
import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from rasterio.enums import Resampling
from pyproj import Transformer
from shapely.geometry import LineString
from scipy.ndimage import gaussian_filter

COG_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN='TRUE',
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
    GDAL_CACHEMAX='256'
)

def meters_per_pixel(transform):
    """Return (xres_m, yres_m) for a north-up projected raster (e.g., UTM)."""
    return abs(transform.a), abs(transform.e)

def gaussian_filter_nan(arr, sigma_pix):
    """
    Gaussian filter that respects NaNs:
    filter(data*mask)/filter(mask)
    """
    data = np.array(arr, dtype=np.float32)
    mask = np.isfinite(data).astype(np.float32)
    data = np.where(np.isfinite(data), data, 0.0)
    num = gaussian_filter(data * mask, sigma=sigma_pix, mode='nearest')
    den = gaussian_filter(mask, sigma=sigma_pix, mode='nearest')
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    return out

def densify_line_xy(xy, step_m):
    """
    Densify a polyline in projected meters so consecutive points are ~step_m apart.
    xy: list[(x,y)] in raster CRS (meters for UTM)
    """
    ls = LineString(xy)
    if ls.length <= 0:
        return xy
    n = max(2, int(math.ceil(ls.length / step_m)) + 1)
    dists = np.linspace(0.0, ls.length, n)
    return [ls.interpolate(d).coords[0] for d in dists]

def length_weights_xy(xy):
    """Return per-point segment length weights (meters) for length-weighted averaging."""
    if len(xy) < 2:
        return np.array([1.0], dtype=np.float32)
    segs = [math.hypot(x2-x1, y2-y1) for (x1,y1),(x2,y2) in zip(xy[:-1], xy[1:])]
    return np.array(segs + [0.0], dtype=np.float32)

# --------- main API ---------

def mechanical_green_score(
    url_frac_cog: str,
    route_lonlat: list,          # [(lon,lat), ...] in EPSG:4326
    weights = {"tree":0.6, "shrub":0.25, "grass":0.15},
    band_map = {"tree":1, "shrub":2, "grass":3},  # 1-based band indices in your fractions COG
    sigma_m: float = 150.0,       # Gaussian kernel σ in meters (radius ~ 3σ)
    pad_m: float = 600.0,         # pad window around route (>= 3σ recommended)
    step_m: float = 25.0,         # densification step along path (m)
    overview_scale: int = 1,      # >1 to downsample via overviews (keeps transfers tiny)
):
    """
    Compute MGS in [0,1] for a route given a multi-band fractions COG (tree/shrub/grass).

    Returns: dict(score=..., samples=..., transform=..., crs=...)
    """
    # 1) Open COG; get CRS (should be UTM meters) and transform
    with rio.Env(**COG_ENV), rio.open(url_frac_cog) as ds:
        crs = ds.crs
        if not crs or not crs.is_projected:
            raise RuntimeError("Fractions COG must be in a projected CRS (e.g., UTM).")
        # Transform route to dataset CRS (meters)
        T = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        route_xy = [T.transform(lon, lat) for lon, lat in route_lonlat]
        # 2) Compute a bbox in dataset CRS with pad
        xs, ys = zip(*route_xy)
        w, s, e, n = min(xs)-pad_m, min(ys)-pad_m, max(xs)+pad_m, max(ys)+pad_m

        # 3) Read only the window we need (bands for tree/shrub/grass)
        win = from_bounds(w, s, e, n, ds.transform).intersection(
            rio.windows.Window(0, 0, ds.width, ds.height)
        )
        if win.width <= 0 or win.height <= 0:
            raise ValueError("Route bbox has no overlap with raster.")
        idxs = [band_map[k] for k in ("tree","shrub","grass")]
        out_h = int(math.ceil(win.height / overview_scale))
        out_w = int(math.ceil(win.width  / overview_scale))
        frac = ds.read(indexes=idxs, window=win, out_shape=(len(idxs), out_h, out_w),
                       resampling=Resampling.average)  # fractions are continuous
        sub_transform = ds.window_transform(win).scaled(win.width/out_w, win.height/out_h)

    # 4) Combine to a single green fraction grid
    tree, shrub, grass = frac.astype(np.float32)
    G = (weights["tree"]*tree +
         weights["shrub"]*shrub +
         weights["grass"]*grass).astype(np.float32)

    # 5) Convolve with Gaussian kernel (meters => pixels)
    px_x, px_y = meters_per_pixel(sub_transform)
    # use isotropic sigma in pixels (average the resolutions)
    px = 0.5*(px_x + px_y)
    sigma_pix = max(0.1, float(sigma_m / px))
    G_smooth = gaussian_filter_nan(G, sigma_pix)

    # 6) Densify route in raster CRS and sample length-weighted average
    route_xy_dense = densify_line_xy(route_xy, step_m=step_m)
    weights_m = length_weights_xy(route_xy_dense)[:, None]  # [N,1]
    # Convert XY to row/col in subwindow
    rc = [rowcol(sub_transform, x, y) for x, y in route_xy_dense]
    rc = [(max(0, min(G_smooth.shape[0]-1, r)),
           max(0, min(G_smooth.shape[1]-1, c))) for r, c in rc]
    vals = np.array([G_smooth[r, c] for r, c in rc], dtype=np.float32)[:, None]
    # Length-weighted mean ignoring NaNs
    mask = np.isfinite(vals).astype(np.float32)
    num = np.nansum(vals * weights_m * mask, axis=0)
    den = np.nansum(weights_m * mask, axis=0) + 1e-9
    score = float(num/den)  # 0..1

    return {
        "score": score,                 # 0..1 (multiply by 100 for %)
        "samples": vals.squeeze(1),     # per-sample values along the route
        "transform": sub_transform,
        "crs": crs
    }
