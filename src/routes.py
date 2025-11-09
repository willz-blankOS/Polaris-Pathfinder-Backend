import math
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS

# OSMnx v2 sane defaults
ox.settings.use_cache = True
ox.settings.overpass_rate_limit = True
ox.settings.timeout = 180

def candidate_routes(
    origin: tuple[int, int],
    dest: tuple[int, int],
    max_routes: int = 16,
    time_slack_min: float = 5.0,
    max_overlap = 0.7,
    walk_speed_mps: float = 1.4, #~5 km/h
    corridor_half_width_m: float = 900.0,
    fallback_pad_m: float = 1200.0
) -> list:
    """
    Finds K fastest routes between origin and destination.
    
    Parameters
    ----------
    origin: tuple
        A tuple representing the (latitude, longitude) of the origin point.
    dest: tuple
        A tuple representing the (latitude, longitude) of the destination point.
    k: int
        The number of fastest routes to find.

    Returns
    -------
    list
        A list of routes, where each route is represented as a list of (longitude, latitude) tuples.
    """
    
    def _utm_epsg(lon, lat):
        zone = int(math.floor((lon + 180) / 6) + 1)
        return int(f"{326 if lat >= 0 else 327}{zone:02d}")
    
    def bbox_pad(origin, dest, pad_m):
        (lon1, lat1) = origin
        (lon2, lat2) = dest
        
        latc = 0.5 * (lat1 + lat2)
        m_per_deg_lat = 111_320
        m_per_deg_lon = 111_320 * math.cos(math.radians(latc))
        dlat = pad_m / m_per_deg_lat
        dlon = pad_m / m_per_deg_lon
        west, east = min(lon1, lon2) - dlon, max(lon1, lon2) + dlon
        south, north = min(lat1, lat2) - dlat, max(lat1, lat2) + dlat
        return (west, south, east, north)
    
    def route_edges(route_nodes):
        return {
            tuple(sorted((u, v)))
            for u, v in zip(route_nodes[:-1], route_nodes[1:])
        }
        
    def route_stats(G, route_nodes):
        dist = 0.0
        time_s = 0.0
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            d = G.get_edge_data(u, v, 0)
            L = float(d.get('length', 0.0))
            dist += L
            time_s += L / walk_speed_mps
        return dist / 1000.0, time_s / 60.0  # km, min

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

    def route_linestring(G, route_nodes):
        coords = []
        for i, (u, v) in enumerate(zip(route_nodes[:-1], route_nodes[1:])):
            d = G.get_edge_data(u, v, 0)
            geom = d.get('geometry')
            pts = list(geom.coords) if geom is not None else [
                (G.nodes[u]["x"], G.nodes[u]["y"]), 
                (G.nodes[v]["x"], G.nodes[v]["y"])
            ]
            
            if i > 0 and coords and pts[0] == coords[-1]:
                coords.extend(pts[1:])
            else:
                coords.extend(pts)
        
        return {'type': 'LineString', 'coordinates': coords}

    G = None
    for hw in (corridor_half_width_m, corridor_half_width_m*1.5):
        poly = _corridor_polygon(origin, dest, hw)
        G = ox.graph_from_polygon(poly, network_type="walk", simplify=True,
                                  retain_all=True, truncate_by_edge=True)
        if len(G) > 0:
            break
    if G is None or len(G) == 0:
        # very small bbox fallback
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

    # 2) edge time attribute (no lambda)
    for _, _, _, d in G.edges(keys=True, data=True):
        L = float(d.get("length", 0.0))
        d["time_s"] = L / walk_speed_mps if L > 0 else 0.0
    
    src = ox.distance.nearest_nodes(G, origin[0], origin[1])
    dst = ox.distance.nearest_nodes(G, dest[0], dest[1])
    
    fastest = nx.shortest_path(G, src, dst, weight="time_s")
    fastest_dist_km, fastest_time_min = _route_stats(G, fastest, walk_speed_mps)
    cap_s = (fastest_time_min + time_slack_min) * 60.0

    # 5) near-fastest, diverse
    chosen, chosen_sets = [], []
    for nodes in nx.shortest_simple_paths(G, src, dst, weight="time_s"):
        dist_km, time_min = _route_stats(G, nodes, walk_speed_mps)
        if time_min*60.0 > cap_s:
            break
        e_set = _route_edges(nodes)
        if any((len(e_set & prev)/max(1,len(e_set|prev))) > max_overlap for prev in chosen_sets):
            continue
        chosen.append({
            "nodes": nodes,
            "time_min": time_min,
            "dist_km": dist_km,
            "geometry": _route_linestring(G, nodes),
        })
        chosen_sets.append(e_set)
        if len(chosen) >= max_routes:
            break

    if not chosen or chosen[0]["nodes"] != fastest:
        chosen.insert(0, {
            "nodes": fastest,
            "time_min": fastest_time_min,
            "dist_km": fastest_dist_km,
            "geometry": _route_linestring(G, fastest),
        })
        if len(chosen) > max_routes:
            chosen = chosen[:max_routes]

    chosen.sort(key=lambda r: r["time_min"])
    return chosen


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