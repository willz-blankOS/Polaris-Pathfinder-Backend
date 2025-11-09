import math
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS

ox.settings.use_cache = True
ox.settings.overpass_rate_limit = True
ox.settings.timeout = 180
# Optional:
# ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"

def _utm_epsg(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return int(f"{326 if lat >= 0 else 327}{zone:02d}")

def _corridor_polygon(origin_ll, dest_ll, half_width_m=900.0):
    (lon1, lat1), (lon2, lat2) = origin_ll, dest_ll
    lonc, latc = (lon1 + lon2)/2.0, (lat1 + lat2)/2.0
    crs_utm = CRS.from_epsg(_utm_epsg(lonc, latc))
    to_utm   = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True).transform
    line_ll  = LineString([(lon1, lat1), (lon2, lat2)])
    poly_utm = shp_transform(to_utm, line_ll).buffer(half_width_m, cap_style=2, join_style=2)
    return shp_transform(to_wgs84, poly_utm)

def _ensure_edge_geometry(G):
    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" not in d:
            d["geometry"] = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                        (G.nodes[v]["x"], G.nodes[v]["y"])])

def _route_stats(G, nodes, walk_speed_mps):
    dist = 0.0; time_s = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        # MultiDiGraph: may have multiple parallel edges; the path algorithm
        # used the minimum-weight one. For stats, take the min time_s across keys.
        dd = G.get_edge_data(u, v) or {}
        best = None
        for k, d in dd.items():
            L = float(d.get("length", 0.0))
            t = L / walk_speed_mps if L > 0 else 0.0
            if best is None or t < best:
                best = t; L_best = L
        if best is None:
            continue
        dist += L_best
        time_s += best
    return dist/1000.0, time_s/60.0

def _route_linestring(G, nodes):
    coords = []
    for i, (u, v) in enumerate(zip(nodes[:-1], nodes[1:])):
        # pick the geometry from the min-time edge between u,v (same logic as above)
        dd = G.get_edge_data(u, v) or {}
        best_k, best_geom, best_time = None, None, None
        for k, d in dd.items():
            geom = d.get("geometry")
            L = float(d.get("length", 0.0))
            t = L  # proportional proxy; we just need a consistent min
            if best_time is None or t < best_time:
                best_time = t; best_k = k; best_geom = geom
        if best_geom is not None:
            pts = list(best_geom.coords)
        else:
            pts = [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        if i>0 and coords and pts[0]==coords[-1]:
            coords.extend(pts[1:])
        else:
            coords.extend(pts)
    return {"type":"LineString","coordinates":coords}

def _jaccard_edges(G, nodes_a, nodes_b):
    def edge_set(nodes):
        s = set()
        for u, v in zip(nodes[:-1], nodes[1:]):
            # Collapse all parallel edges to the unordered pair
            s.add(tuple(sorted((u, v))))
        return s
    A, B = edge_set(nodes_a), edge_set(nodes_b)
    return len(A & B) / max(1, len(A | B))

def candidate_routes(
    origin, dest,
    max_routes=8,
    time_slack_min=5.0,
    max_overlap=0.70,
    walk_speed_mps=1.4,
    corridor_half_width_m=900.0,
    fallback_pad_m=1200.0,
    penalty_sec=60.0,          # per-used-edge penalty added to time_s
    max_tries=40,
):
    """
    Returns diversified, near-fastest pedestrian routes without using
    nx.shortest_simple_paths (works with MultiDiGraph).
    origin/dest are (lon, lat).
    """
    # 1) fetch graph via corridor (then small bbox fallback)
    G = None
    for hw in (corridor_half_width_m, corridor_half_width_m*1.5):
        poly = _corridor_polygon(origin, dest, hw)
        G = ox.graph_from_polygon(poly, network_type="walk", simplify=True,
                                  retain_all=True, truncate_by_edge=True)
        if len(G) > 0:
            break
    if G is None or len(G) == 0:
        (lon1, lat1), (lon2, lat2) = origin, dest
        latc = 0.5*(lat1+lat2)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0*math.cos(math.radians(latc))
        dlat = fallback_pad_m/m_per_deg_lat
        dlon = fallback_pad_m/m_per_deg_lon
        bbox = (min(lon1,lon2)-dlon, min(lat1,lat2)-dlat, max(lon1,lon2)+dlon, max(lat1,lat2)+dlat)
        G = ox.graph_from_bbox(bbox=bbox, network_type="walk",
                               simplify=True, retain_all=True, truncate_by_edge=True)
        if len(G) == 0:
            raise RuntimeError("No OSM data found (check coords or widen corridor).")

    _ensure_edge_geometry(G)

    # 2) base edge time
    for u, v, k, d in G.edges(keys=True, data=True):
        L = float(d.get("length", 0.0))
        d["time_base"] = L / walk_speed_mps if L > 0 else 0.0
        d["time_pen"]  = 0.0

    # helper: compute effective time_s = base + penalty
    def _compute_time_s(G_):
        for _, _, _, d in G_.edges(keys=True, data=True):
            d["time_s"] = float(d.get("time_base", 0.0)) + float(d.get("time_pen", 0.0))

    _compute_time_s(G)

    # 3) snap endpoints
    src = ox.distance.nearest_nodes(G, origin[0], origin[1])
    dst = ox.distance.nearest_nodes(G, dest[0],  dest[1])

    # 4) fastest path + cap
    fastest = nx.shortest_path(G, src, dst, weight="time_s")
    fastest_dist_km, fastest_time_min = _route_stats(G, fastest, walk_speed_mps)
    cap_s = (fastest_time_min + time_slack_min) * 60.0

    routes = [fastest]
    tries  = 0

    # 5) iteratively penalize used edges and recompute new alt paths
    while len(routes) < max_routes and tries < max_tries:
        tries += 1
        # Apply penalty to ALL parallel edges between each pair in the last route
        last = routes[-1]
        for u, v in zip(last[:-1], last[1:]):
            for k, d in (G.get_edge_data(u, v) or {}).items():
                d["time_pen"] = d.get("time_pen", 0.0) + penalty_sec
        _compute_time_s(G)

        try:
            cand = nx.shortest_path(G, src, dst, weight="time_s")
        except nx.NetworkXNoPath:
            break

        # time cap
        _, cand_time_min = _route_stats(G, cand, walk_speed_mps)
        if cand_time_min*60.0 > cap_s:
            break

        # diversity vs all existing routes
        if any(_jaccard_edges(G, cand, r) > max_overlap for r in routes):
            # increase penalty and try again
            penalty_sec *= 1.25
            continue

        routes.append(cand)

    # 6) package outputs
    out = []
    for nodes in routes:
        dist_km, time_min = _route_stats(G, nodes, walk_speed_mps)
        out.append({
            "nodes": nodes,
            "time_min": time_min,
            "dist_km": dist_km,
            "geometry": _route_linestring(G, nodes),
        })

    # sort by time and enforce cap (again)
    out = sorted([r for r in out if r["time_min"]*60.0 <= cap_s], key=lambda r: r["time_min"])
    return out

def topk_routes(
    origin: tuple[int, int],
    dest: tuple[int, int],
    k: int = 5,
    **kwargs
) -> list:
    """
    Finds the top K fastest routes between origin and destination.
    
    Parameters
    ----------
    origin: tuple
        A tuple representing the (latitude, longitude) of the origin point.
    dest: tuple
        A tuple representing the (latitude, longitude) of the destination point.
    k: int
        The number of fastest routes to return.

    Returns
    -------
    list
        A list of the top K fastest routes, where each route is represented as a list of (longitude, latitude) tuples.
    """
    routes = candidate_routes(origin, dest, max_routes=k, **kwargs)
    
    # TODO: Rank routes by combined risk score
    
    return routes[:k]