import osmnx as ox
import networkx as nx
import numpy as np
from functools import lru_cache

print("Loading Pasay road network...")
try:
    G = ox.graph_from_bbox(bbox=(14.570, 14.505, 121.040, 120.975),network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"Road netwwork loadedd: {len(G.nodes)} nodes, {len(G.edges)} edges")
except TypeError:
    try:
        G = ox.graph_from_bbox(14.570, 14.505, 121.040, 120.975, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        print(f"Rooad network loaded (legacy API): {len(G.nodes)} nodes, {len(G.edges)} edges")
    except Exception as e2:
        print(f"Failed to loadd road network: {e2}")
except Exception as e:
    print(f"Failed  to load road network: {e}")
    G = None
    
# Crime Penalty Lookup

def get_crime_penalty(lat, lgn, heatmap_points):
    """Return the crime penalty of the nearest barangay to a given point"""
    if not heatmap_points:
        return 25.0

    best_penalty = 25.0
    best_dist = float('inf')

    for b in heatmap_points:
        dlat = lat - b['lat']
        dlng = lat - b['lng']
        dist = dlat * dlat + dlng * dlng

        if dist < best_dist:
            best_dist = dist
            best_penalty = b['crime_penalty']

    return best_penalty


def build_weighted_graph(heatmap_points, lambda_weight=0.5):
    """
    Returns a copy of G where each edge weight implements the thesis formula:
        w'(u,v) = travel_time(u,v) * (1 + λ * normalised_crime_penalty)
 
    This means high-crime roads cost more to traverse, so A* naturally
    routes around them.
 
    lambda_weight controls the crime vs speed trade-off:
        0.0 = pure fastest route (ignore crime)
        0.5 = balanced (default)
        1.5 = strongly avoid crime even at significant time cost
    """
    if G is None:
        return None

    penalties = [b['crime_penalty'] for b in heatmap_points] if heatmap_points else [25]
    max_p = max(penalties) or 1
    min_p = min(penalties) or 0

    H = G.copy()

    for u, v, key, data, in H.edges(data=True, keys=True):
        u_data = H.nodes[u]
        v_data = H.nodes[v]
        mid_lat = (u_data['y'] + v_data['y']) / 2
        mid_lng = (u_data['x'] + v_data['x']) / 2

        raw_penalty = get_crime_penalty(mid_lat, mid_lng, heatmap_points)
        norm_penalty = (raw_penalty - min_p) / (max_p - min_p) if max_p > min_p else 0

        # Modified edge weight: w'(u,v) = travel_time * (1 + λ * crime_penalty)

        base_weight = data.get('travel_time', data.get('length', 1))
        data['safe_weight'] = base_weight * (1 + lambda_weight * norm_penalty)

        return H


def find_safe_route(origin_lat, origin_lng, dest_lat, dest_lng, heatmap_points, lambda_weight=0.5):
    """
    Find the crime-weighted safest route between two points.
 
    Returns:
        {
            'polyline':  [[lat,lng], ...],
            'distance':  metres (float),
            'duration':  seconds (float),
            'crime_cost': total crime-weighted cost
        }
    """
    if G is None:
        raise RuntimeError("Road network not loaded")

    # Snap origin and destination to nearest OSM nodes
    orig_node = ox.nearest_nodes(G, X=origin_lng, Y=origin_lat)
    dest_node = ox.nearest_nodes(G, X=  dest_lng, Y=dest_lat)

    if orig_node == dest_node:
        raise ValueError("Origin and destination are the same node")

    # Build crime_weighted graph
    H = build_weighted_graph(heatmap_points, lambda_weight)

    # Run A* witth crime_weighted edges
    # nx.astar_path uses the Haversine heuristic via the weight parameter

    try:
        path = nx.astar_path(H, orig_node, dest_node, weight='safe_weight')
    except nx.NetworkXNoPath:
        # Fallback to shortest path if A* fails

        path = nx.shortest_path(H, orig_node, dest_node, weight='length')

    # Extract polyline from node sequence
    polyline = []
    for node in path:
        node_data = H.nodes[node]
        polyline.append([node_data['y'], node_data['x']])

    # Calculate actual distance and duration along the path

    total_distance = 0
    total_duration = 0
    total_crime_cost = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Gett the best (lowest weight) edge between these nodes

        edge_data = min(
            H[u][v].values(), key=lambda d: d.get('safe_weight', float('inf'))
        )
        total_distance += edge_data.get('length', 0)
        total_duration  += edge_data.get('travel_time', 0)
        total_crime_cost += edge_data.get('safe_weight', 0)

    return {
        'polyline': polyline,
        'distance': round(total_distance, 1),
        'duration': round(total_duration, 1),
        'crime_cost': round(total_crime_cost, 2)
    }

def compute_three_routes(origin_lat, origin_lng, dest_lat, dest_lng, heatmap_points):
    """
    Compute safest, balanced, and fastest routes by varying λ.
 
    λ = 1.5 → strongly avoids crime  (safest)
    λ = 0.5 → balanced               (balanced)
    λ = 0.0 → pure travel time       (fastest)
    """

    configs = [
        {'id': 'safest', 'label': 'Safest Route', 'tag': 'Recommended',
        'desc': 'Avoids high crime_penalty roads.', 'lambda': 1.5},
        {'id': 'balanced', 'label' : 'Balanced Route', 'tag': 'Balanced', 
        'desc': 'Moderate crime avoidance', 'lambda': 0.5},
        {'id': 'fastest', 'label': 'Fastest Route', 'tag': 'Fastest',
        'desc': 'Shortest time, higher crime risk', 'lambda': 0.5}
    ]

    # Precompute penalty stats for scoring
    penalties = [b['crime_penalty'] for b in heatmap_points] if heatmap_points else [25]
    max_p = max(penalties) or 1

    results = []
    for cfg in configs:
        route = find_safe_route(
            origin_lat, origin_lng, dest_lat, dest_lng, 
            heatmap_points, lambda_weight=cfg['lambda']    
        )

    # Safety score: inverse of normalised crime cost
    # lower crime_cost = higher safety score

        norm_cost = min(1.0, route['crime_cost'] / (route['distance'] * 2 + 1))
        score = round(max(40, min(95, 95 - norm_cost * 40)))

        mins = round(route['duration'] / 60)
        dur = f"{mins} min" if mins < 60 else f"{mins//60}h {mins%60}m"
        dist = f"{route['distance']/1000:1f} km"  if route['distance'] >= 1000 \
            else f"{round(route['distance'])} m"

        color  = '#2D6A4F' if score >= 80 else '#EF8C2D' if score >= 60 else '#D62828'
        tag_bg = '#EBF5F0' if score >= 80 else '#FFF4E6' if score >= 60 else '#FDEAEA'
    
        results.append({
            **cfg,
            'score':      score,
            'scoreColor': color,
            'tagBg':      tag_bg,
            'tagColor':   color,
            'duration':   dur,
            'distance':   dist,
            'polyline':   route['polyline'],
            'crime_cost': route['crime_cost'],
        })

    return results