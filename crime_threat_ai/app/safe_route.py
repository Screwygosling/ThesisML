# safe_route.py
# Crime-weighted A* routing using OSM road network + street-level incident data
# Penalty index pre-computed at startup to avoid OOM on Render free tier

import networkx as nx
import math
import json
import os

BUNDLED_FILE   = os.path.join(os.path.dirname(__file__), 'pasay_roads.json')
TEMP_CACHE     = '/tmp/pasay_graph.json'
INCIDENTS_FILE = os.path.join(os.path.dirname(__file__), 'crime_incidents.json')

# ── Haversine ─────────────────────────────────────────────────────────────────
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

# ── Build graph from Overpass JSON ────────────────────────────────────────────
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
        speed   = speed_map.get(tags.get('highway', 'residential'), 25)
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
    return None

# ── Load incidents ────────────────────────────────────────────────────────────
def load_incidents():
    if os.path.exists(INCIDENTS_FILE):
        try:
            with open(INCIDENTS_FILE) as f:
                data = json.load(f)
            print(f"Crime incidents loaded: {len(data)} records")
            return data
        except Exception as e:
            print(f"Failed to load incidents: {e}")
    return []

print("Loading Pasay road network...")
G = load_graph()
if G is None:
    print("Road network unavailable")

print("Loading crime incident data...")
INCIDENTS = load_incidents()

# ── Pre-compute penalty index at startup ──────────────────────────────────────
# Runs once when server starts — O(nodes x incidents)
# Avoids recomputing on every request which causes OOM on Render free tier
PENALTY_INDEX = {}
PENALTY_MIN   = 40.0
PENALTY_MAX   = 40.0

def precompute_penalty_index():
    global PENALTY_INDEX, PENALTY_MIN, PENALTY_MAX
    if G is None or not INCIDENTS:
        print("Skipping penalty pre-computation")
        return

    print(f"Pre-computing penalty index ({G.number_of_nodes()} nodes x {len(INCIDENTS)} incidents)...")
    index = {}
    for node, ndata in G.nodes(data=True):
        best_p = 0.0
        for inc in INCIDENTS:
            dlat = ndata['lat'] - inc['lat']
            dlng = ndata['lng'] - inc['lng']
            d2   = dlat * dlat + dlng * dlng
            # 200m radius in degrees squared ~ 0.0000032
            if d2 < 0.0000032:
                # Weight by proximity — closer incidents contribute more
                weight  = 1.0 - (d2 / 0.0000032)
                best_p += inc['crime_penalty'] * weight
        index[node] = min(100.0, best_p)  # cap at 100

    PENALTY_INDEX = index
    vals          = list(index.values())
    PENALTY_MIN   = min(vals)
    PENALTY_MAX   = max(vals)
    print(f"Penalty index ready. Range: {PENALTY_MIN:.1f} - {PENALTY_MAX:.1f}")
    high = sum(1 for v in vals if v > 50)
    print(f"High-crime nodes (>50): {high} of {len(vals)}")

precompute_penalty_index()

# ── Build crime-weighted graph ────────────────────────────────────────────────
def build_weighted_graph(lambda_weight=0.5):
    if G is None:
        return None

    rng = (PENALTY_MAX - PENALTY_MIN) or 1
    H   = G.copy()

    for u, v, data in H.edges(data=True):
        pu           = PENALTY_INDEX.get(u, 0.0)
        pv           = PENALTY_INDEX.get(v, 0.0)
        raw_penalty  = (pu + pv) / 2
        norm_penalty = (raw_penalty - PENALTY_MIN) / rng
        base         = data.get('travel_time', data.get('length', 1))
        # Thesis formula: w'(u,v) = travel_time * (1 + lambda * crime_penalty)
        data['safe_weight'] = base * (1 + lambda_weight * norm_penalty)

    return H

# ── Cache weighted graphs at startup too ──────────────────────────────────────
# Build all three lambda variants once so requests are instant
print("Pre-building weighted graphs...")
H_SAFE     = build_weighted_graph(lambda_weight=1.5) if G else None
H_BALANCED = build_weighted_graph(lambda_weight=0.5) if G else None
H_FASTEST  = build_weighted_graph(lambda_weight=0.0) if G else None
print("Weighted graphs ready.")

# ── Nearest node ──────────────────────────────────────────────────────────────
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

# ── Find route on a pre-built graph ──────────────────────────────────────────
def find_route_on_graph(H, orig_node, dest_node):
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
def compute_three_routes(origin_lat, origin_lng, dest_lat, dest_lng, heatmap_points=None):
    if G is None or H_SAFE is None:
        raise RuntimeError("Road network not loaded")

    orig_node = nearest_node(origin_lat, origin_lng)
    dest_node = nearest_node(dest_lat, dest_lng)

    if orig_node == dest_node:
        raise ValueError("Origin and destination map to the same node")

    # Use pre-built graphs — no computation needed per request
    safe_route     = find_route_on_graph(H_SAFE,     orig_node, dest_node)
    balanced_route = find_route_on_graph(H_BALANCED, orig_node, dest_node)
    fastest_route  = find_route_on_graph(H_FASTEST,  orig_node, dest_node)

    def fmt_time(s):
        m = round(s / 60)
        return f"{m} min" if m < 60 else f"{m//60}h {m%60}m"

    def fmt_dist(m):
        return f"{m/1000:.1f} km" if m >= 1000 else f"{round(m)} m"

    def score(route):
        norm = min(1.0, route['crime_cost'] / (route['distance'] * 2 + 1))
        return round(max(40, min(95, 95 - norm * 40)))

    def color(s):  return '#2D6A4F' if s >= 80 else '#EF8C2D' if s >= 60 else '#D62828'
    def tag_bg(s): return '#EBF5F0' if s >= 80 else '#FFF4E6' if s >= 60 else '#FDEAEA'

    results = []
    for cfg, route in [
        {'id':'safest',   'label':'Safest Route',   'tag':'Recommended', 'desc':'Avoids roads near recorded crime incidents.'},
        {'id':'balanced', 'label':'Balanced Route', 'tag':'Balanced',    'desc':'Moderate crime avoidance.'},
        {'id':'fastest',  'label':'Fastest Route',  'tag':'Fastest',     'desc':'Shortest time, higher crime risk.'},
    ], [safe_route, balanced_route, fastest_route]:
        s = score(route)
        results.append({
            **cfg,
            'score':      s,
            'scoreColor': color(s),
            'tagBg':      tag_bg(s),
            'tagColor':   color(s),
            'duration':   fmt_time(route['duration']),
            'distance':   fmt_dist(route['distance']),
            'polyline':   route['polyline'],
            'crime_cost': route['crime_cost'],
        })

    return results