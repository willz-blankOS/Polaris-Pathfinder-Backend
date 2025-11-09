import math
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString

def candidate_routes(
    origin: tuple[int, int],
    dest: tuple[int, int],
    max_routes: int = 16,
    time_slack_min: float = 5.0,
    max_overlap = 0.7,
    pad_m: float = 2000.0, #(meters)
    walk_speed_mps: float = 1.4, #~5 km/h
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
        return (north, south, east, west)
    
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

    def ensure_edge_geometry(G):
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'geometry' not in data:
                point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
                point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
                data['geometry'] = LineString([point_u, point_v])

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

    north, south, east, west = bbox_pad(origin, dest, pad_m)
    G = ox.graph_from_bbox(
        (west, south, east, north), network_type='walk', simplify=True
    )
    ensure_edge_geometry(G)
    
    src = ox.distance.nearest_nodes(G, origin[0], origin[1])
    dst = ox.distance.nearest_nodes(G, dest[0], dest[1])
    
    fastest_route = ox.shortest_path(
        G, src, dst, 
        weight=lambda u,v,k,d: d.get('length', 0.0)/walk_speed_mps
    )
    fastest_dist_km, fastest_time_min = route_stats(G, fastest_route)
    cap_s = (fastest_time_min + time_slack_min) * 60.0  # seconds
    
    chosen = []
    chosen_edge_sets = []
    
    gen = nx.shortest_simple_paths(
        G, src, dst, 
        weight=lambda u,v,k,d: d.get('length', 0.0)/walk_speed_mps
    )
    for nodes in gen:
        dist_km, time_min = route_stats(G, nodes)
        if time_min * 60.0 > cap_s:
            break
        
        edge_set = route_edges(nodes)
        is_diverse = True
        for prev in chosen_edge_sets:
            jacc = len(edge_set & prev) / max(1, len(edge_set | prev))
            if jacc > max_overlap:
                is_diverse = False
                break
        if not is_diverse:
            continue
        
        chosen.append({
            "nodes": nodes,
            "time_min": time_min,
            "dist_km": dist_km,
            "geometry": route_linestring(G, nodes)
        })
        chosen_edge_sets.append(edge_set)
        if len(chosen) >= max_routes:
            break
        
    if not chosen or chosen[0]["nodes"] != fastest_route:
        chosen.insert(0, {
            "nodes": fastest_route,
            "time_min": fastest_time_min,
            "dist_km": fastest_dist_km,
            "geometry": route_linestring(G, fastest_route)
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