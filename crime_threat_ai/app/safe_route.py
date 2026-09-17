# safe_route.py
# Crime-weighted A* routing using OSM road network + street-level incident data

import networkx as nx
import requests
import math
import json
import os

BUNDLED_FILE    = os.path.join(os.path.dirname(__file__), 'pasay_roads.json')
TEMP_CACHE      = '/tmp/pasay_graph.json'
INCIDENTS_FILE  = os.path.join(os.path.dirname(__file__), 'crime_incidents.json')

# ── Haversine distance in metres ──────────────────────────────────────────────
def haversine(a, b):
    R    = 6371000
    dLat = math.radians(b[0] - a[0])
    dLng = math.radians(b[1] - a[1])
    s    = (math.sin(dLat/2)**2 +
            math.cos(math.radians(a[0])) *
            math.cos(math.radians(b[0])) *
            math.sin(dLng/2)**2)
    s = max(0.0, min(1.0, s))
    return R * 2 * math.atan2(math.sqrt(s), math.sqrt(1 - s))

# ── Build NetworkX graph from Overpass JSON ───────────────────────────────────
def build_graph(overpass_data):
    G = nx.DiGraph()
    nodes = {}
    for el in overpass_data.get('elements', []):
        if el['type'] == 'node':
            nodes[el['id']] = (el['lat'], el['lon'])
            G.add_node(el['id'], lat=el['lat'], lng=el['lon'])

    speed_map = {
        'motorway': 90, 'trunk': 70, 'primary': 50,
        'secondary': 40, 'tertiary': 30, 'residential': 20,
        'unclassified': 20, 'living_street': 10,
    }

    for el in overpass_data.get('elements', []):
        if el['type'] != 'way':
            continue
        refs    = el.get('nodes', [])
        tags    = el.get('tags', {})
        oneway  = tags.get('oneway', 'no') == 'yes'
        highway = tags.get('highway', 'residential')
        speed   = speed_map.get(highway, 25)

        for i in range(len(refs) - 1):
            u, v = refs[i], refs[i+1]
            if u not in nodes or v not in nodes:
                continue
            dist  = haversine(nodes[u], nodes[v])
            ttime = (dist / 1000) / speed * 3600
            G.add_edge(u, v, length=dist, travel_time=ttime)
            if not oneway:
                G.add_edge(v, u, length=dist, travel_time=ttime)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ── Load road network ─────────────────────────────────────────────────────────
def load_graph():
    for path, label in [(BUNDLED_FILE, 'bundled file'), (TEMP_CACHE, 'temp cache')]:
        if os.path.exists(path):
            try:
                print(f"Loading road network from {label}...")
                with open(path) as f:
                    data = json.load(f)
                G = build_graph(data)
                if G.number_of_nodes() > 0:
                    print(f"Road network loaded: {G.number_of_nodes()} nodes")
                    return G
            except Exception as e:
                print(f"Load from {label} failed: {e}")
    print("No road network available")
    return None

# ── Load crime incidents ──────────────────────────────────────────────────────
def load_incidents():
    if os.path.exists(INCIDENTS_FILE):
        try:
            with open(INCIDENTS_FILE) as f:
                incidents = json.load(f)
            print(f"Crime incidents loaded: {len(incidents)} records")
            return incidents
        except Exception as e:
            print(f"Failed to load incidents: {e}")
    print("No incident file found -- falling back to heatmap points")
    return []

print("Loading Pasay road network...")
G = load_graph()
if G is None:
    print("Road network unavailable -- /safe-route will return 503")

print("Loading crime incident data...")
INCIDENTS = load_incidents()

# ── Street-level crime penalty lookup ────────────────────────────────────────
# Uses actual incident coordinates with a tight influence radius (200m)
# so only roads that genuinely pass near a crime scene are penalized.
# Falls back to heatmap barangay data if no incident file is available.
def get_incident_penalty(lat, lng, incidents, heatmap_points, radius=200):
    total_penalty = 0.0

    if incidents:
        # Sum weighted penalties from all incidents within radius
        # Closer incidents contribute more (inverse distance weighting)
        for inc in incidents:
            d = haversine((lat, lng), (inc['lat'], inc['lng']))
            if d < radius:
                weight         = 1.0 - (d / radius)  # 1.0 at same spot, 0 at radius
                total_penalty += inc['crime_penalty'] * weight

        # Normalise to 0-100 scale
        # A single severity-5 incident at 0m contributes 100 * 1.0 = 100
        # Multiple incidents accumulate — cap at 100
        return min(100.0, total_penalty)

    # Fallback: use heatmap barangay data
    if heatmap_points:
        best_p = 40.0
        best_d = float('inf')
        for b in heatmap_points:
            d = haversine((lat, lng), (b['lat'], b['lng']))
            if d < best_d:
                best_d = d
                best_p = b['crime_penalty']
        return best_p if best_d < 600 else 40.0

    return 40.0  # neutral fallback

# ── Pre-compute penalty index for all graph nodes ────────────────────────────
# Done once per request — O(nodes x incidents) instead of O(edges x incidents)
def build_penalty_index(heatmap_points):
    if G is None:
        return {}
    incidents = INCIDENTS  # use bundled incident data
    index = {}
    for node, ndata in G.nodes(data=True):
        index[node] = get_incident_penalty(
            ndata['lat'], ndata['lng'],
            incidents, heatmap_points
        )
    return index

# ── Build crime-weighted graph ────────────────────────────────────────────────
def build_weighted_graph(heatmap_points, lambda_weight=0.5):
    if G is None:
        return None

    # Get penalty range for normalisation
    penalty_index = build_penalty_index(heatmap_points)
    all_penalties = list(penalty_index.values())
    max_p = max(all_penalties) if all_penalties else 100
    min_p = min(all_penalties) if all_penalties else 0
    rng   = (max_p - min_p) or 1

    H = G.copy()
    for u, v, data in H.edges(data=True):
        pu           = penalty_index.get(u, 40.0)
        pv           = penalty_index.get(v, 40.0)
        raw_penalty  = (pu + pv) / 2
        norm_penalty = (raw_penalty - min_p) / rng
        base         = data.get('travel_time', data.get('length', 1))
        # Thesis formula: w'(u,v) = travel_time * (1 + lambda * crime_penalty)
        data['safe_weight'] = base * (1 + lambda_weight * norm_penalty)

    return H

# ── Nearest node lookup ───────────────────────────────────────────────────────
def nearest_node(lat, lng):
    best_node = None
    best_dist = float('inf')
    for node, data in G.nodes(data=True):
        dlat = lat - data['lat']
        dlng = lng - data['lng']
        d2   = dlat * dlat + dlng * dlng
        if d2 < best_dist:
            best_dist = d2
            best_node = node
    return best_node

# ── Find single route ─────────────────────────────────────────────────────────
def find_route(origin_lat, origin_lng, dest_lat, dest_lng,
               heatmap_points, lambda_weight=0.5):
    if G is None:
        raise RuntimeError("Road network not loaded")

    orig_node = nearest_node(origin_lat, origin_lng)
    dest_node = nearest_node(dest_lat, dest_lng)

    if orig_node == dest_node:
        raise ValueError("Origin and destination map to the same node")

    H = build_weighted_graph(heatmap_points, lambda_weight)

    try:
        path = nx.astar_path(
            H, orig_node, dest_node,
            heuristic=lambda u, v: haversine(
                (H.nodes[u]['lat'], H.nodes[u]['lng']),
                (H.nodes[v]['lat'], H.nodes[v]['lng'])
            ),
            weight='safe_weight'
        )
    except nx.NetworkXNoPath:
        path = nx.shortest_path(H, orig_node, dest_node, weight='length')

    polyline   = [[H.nodes[n]['lat'], H.nodes[n]['lng']] for n in path]
    total_dist = total_time = total_crime = 0

    for i in range(len(path) - 1):
        u, v  = path[i], path[i+1]
        edge  = H[u][v]
        total_dist  += edge.get('length', 0)
        total_time  += edge.get('travel_time', 0)
        total_crime += edge.get('safe_weight', 0)

    return {
        'polyline':   polyline,
        'distance':   round(total_dist, 1),
        'duration':   round(total_time, 1),
        'crime_cost': round(total_crime, 2),
    }

# ── Main export ───────────────────────────────────────────────────────────────
def compute_three_routes(origin_lat, origin_lng, dest_lat, dest_lng, heatmap_points):
    configs = [
        {'id': 'safest',   'label': 'Safest Route',   'tag': 'Recommended',
         'desc': 'Avoids roads near recorded crime incidents.', 'lambda': 1.5},
        {'id': 'balanced', 'label': 'Balanced Route', 'tag': 'Balanced',
         'desc': 'Moderate crime avoidance.',                   'lambda': 0.5},
        {'id': 'fastest',  'label': 'Fastest Route',  'tag': 'Fastest',
         'desc': 'Shortest time, higher crime risk.',           'lambda': 0.0},
    ]

    results = []
    for cfg in configs:
        route  = find_route(origin_lat, origin_lng, dest_lat, dest_lng,
                            heatmap_points, lambda_weight=cfg['lambda'])
        norm   = min(1.0, route['crime_cost'] / (route['distance'] * 2 + 1))
        score  = round(max(40, min(95, 95 - norm * 40)))
        mins   = round(route['duration'] / 60)
        dur    = f"{mins} min" if mins < 60 else f"{mins//60}h {mins%60}m"
        dist   = (f"{route['distance']/1000:.1f} km"
                  if route['distance'] >= 1000 else f"{round(route['distance'])} m")
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