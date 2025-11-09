import re
from urllib.parse import unquote

def _to_float_degrees(component: str) -> float:
    s = component.strip().upper()
    
    hem = None
    for h in "NSEW":
        if h in s:
            hem = h
            s = s.replace(h, "")
            
    s = (s
        .replace("DEG", " ")
         .replace("°", " ").replace("º", " ")
         .replace("’", " ").replace("′", " ").replace("'", " ")
         .replace("”", " ").replace("″", " ").replace('"', " ")
    )

    parts = [p for p in re.split(r"[^\d+\-\.]+", s) if p]
    if not parts:
        raise ValueError(f"Could not parse coordinate: {component!r}")

    nums = list(map(float, parts))
    deg = nums[0]
    minutes = nums[1] if len(nums) > 1 else 0.0
    seconds = nums[2] if len(nums) > 2 else 0.0

    val = abs(deg) + minutes / 60.0 + seconds / 3600.0

    if hem in ("S", "W"):
        sign = -1
    elif hem in ("N", "E"):
        sign = 1
    else:
        sign = -1 if deg < 0 else 1

    return sign * val

def parse_lonlat_pair(text: str) -> tuple[float, float]:
    """
    Parse "lon,lat" (recommended) or "lat lon" into (lat, lon) floats.
    Works with decimal or DMS components; use a comma if your DMS contains spaces.
    """
    t = text.strip()
    # Prefer comma; fall back to single whitespace split
    if "," in t:
        a, b = t.split(",", 1)
    else:
        # Only safe if each coord is one token (e.g., decimal) or you pre-split DMS
        parts = t.split()
        if len(parts) != 2:
            raise ValueError(f"Ambiguous coord pair (use a comma): {text!r}")
        a, b = parts

    lat = _to_float_degrees(a)
    lon = _to_float_degrees(b)

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range: {lat} from {text!r}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range: {lon} from {text!r}")

    return lon, lat

# ---- Path parsing helper ----

_GET_ROUTE_RX = re.compile(r"^/(?:api/)?get_route/from=(.+?)&&to=(.+?)$", re.IGNORECASE)

def parse_get_route_path(path: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Parse paths of the form:
      get_route/from={origin_coords}&&to={destination_coords}

    Examples:
      get_route/from=43.651,-79.383&&to=43.7,-79.4
      /get_route/from=N43°39'03\",W79°22'59\"&&to=43.70,-79.40    (URL-encoded in real requests)

    Returns: ((olat, olon), (dlat, dlon))
    """
    # Strip query string if present (we only care about the path)
    path_only = path.split("?", 1)[0]
    m = _GET_ROUTE_RX.match(path_only)
    if not m:
        raise ValueError(f"Path doesn't match get_route format: {path!r}")

    raw_origin = unquote(m.group(1))
    raw_dest = unquote(m.group(2))

    origin = parse_lonlat_pair(raw_origin)
    dest   = parse_lonlat_pair(raw_dest)
    return origin, dest
