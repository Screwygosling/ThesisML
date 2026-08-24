# safe_route.py
# Crime-weighted A* routing using OSM road network via Overpass API
# Bypasses OSMnx geocoding entirely — uses direct Overpass query

import networkx as nx
import requests
import math
import json
import os

# Check multiple cache locations — bundled file first, then temp
BUNDLED_CACHE = os.path.join(os.path.dirname(__file__), 'pasay_roads.json')
TEMP_CACHE    = '/tmp/pasay_graph.json'
CACHE_FILE    = BUNDLED_CACHE if os.path.exists(BUNDLED_CACHE) else TEMP_CACHE

# ── Haversine distance in metres ──────────────────────────────────────────────
def haversine(a, b):
    R = 6371000
    dLat = math.radians(b[0] - a[0])
    dLng = math.radians(b[1] - a[1])
    s = (math.sin(dLat/2)**2 +
         math.cos(math.radians(a[0])) *
         math.cos(math.radians(b[0])) *
         math.sin(dLng/2)**2)
    return R * 2 * math.atan2(math.sqrt(s), math.sqrt(1-s))

# ── Download Pasay road network from Overpass API ────────────────────────────
def download_road_network():
    # Pasay City bounding box: south,west,north,east
    query = """
    [out:json][timeout:60];
    (
      way["highway"]["highway"!~"footway|cycleway|path|pedestrian|steps|service"]
         (14.505,120.975,14.570,121.040);
    );
    out body;
    >;
    out skel qt;
    """
    print("Downloading Pasay road network from Overpass API...")
    try:
        resp = requests.post(
            'https://overpass-api.de/api/interpreter',
            data={'data': query},
            headers={'User-Agent': 'SafeCommutePH/1.0 (thesis research project)'},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Primary Overpass failed: {e}, trying mirror...")
        try:
            resp = requests.post(
                'https://overpass.kumi.systems/api/interpreter',
                data={'data': query},
                headers={'User-Agent': 'SafeCommutePH/1.0 (thesis research project)'},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e2:
            print(f"Mirror also failed: {e2}")
            return None

# ── Build NetworkX graph from Overpass data ───────────────────────────────────
def build_graph(overpass_data):
    G = nx.DiGraph()

    # Index nodes by OSM id
    nodes = {}
    for el in overpass_data.get('elements', []):
        if el['type'] == 'node':
            nodes[el['id']] = (el['lat'], el['lon'])
            G.add_node(el['id'], lat=el['lat'], lng=el['lon'])

    # Add edges from ways
    for el in overpass_data.get('elements', []):
        if el['type'] != 'way':
            continue
        refs = el.get('nodes', [])
        tags = el.get('tags', {})
        oneway = tags.get('oneway', 'no') == 'yes'

        # Estimate speed from highway tag
        highway = tags.get('highway', 'residential')
        speed_map = {
            'motorway': 90, 'trunk': 70, 'primary': 50,
            'secondary': 40, 'tertiary': 30, 'residential': 20,
            'unclassified': 20, 'living_street': 10,
        }
        speed_kph = speed_map.get(highway, 25)

        for i in range(len(refs) - 1):
            u, v = refs[i], refs[i+1]
            if u not in nodes or v not in nodes:
                continue
            dist = haversine(nodes[u], nodes[v])
            travel_time = (dist / 1000) / speed_kph * 3600  # seconds

            G.add_edge(u, v, length=dist, travel_time=travel_time, speed=speed_kph)
            if not oneway:
                G.add_edge(v, u, length=dist, travel_time=travel_time, speed=speed_kph)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ── Load graph (with cache) ───────────────────────────────────────────────────
def load_graph():
    # Try bundled file first (fastest, no network needed)
    if os.path.exists(BUNDLED_CACHE):
        try:
            print("Loading road network from bundled file...")
            with open(BUNDLED_CACHE) as f:
                data = json.load(f)
            G = build_graph(data)
            if G.number_of_nodes() > 0:
                print(f"✅ Road network loaded from bundle: {G.number_of_nodes()} nodes")
                return G
        except Exception as e:
            print(f"Bundle load failed: {e}")

    # Try temp cache (set by a previous worker)
    if os.path.exists(TEMP_CACHE):
        try:
            print("Loading road network from temp cache...")
            with open(TEMP_CACHE) as f:
                data = json.load(f)
            G = build_graph(data)
            if G.number_of_nodes() > 0:
                print(f"✅ Road network loaded from cache: {G.number_of_nodes()} nodes")
                return G
        except Exception as e:
            print(f"Cache load failed: {e}")

    # Wait up to 30s for another worker to populate the cache
    import time
    for _ in range(6):
        time.sleep(5)
        if os.path.exists(TEMP_CACHE):
            try:
                with open(TEMP_CACHE) as f:
                    data = json.load(f)
                G = build_graph(data)
                if G.number_of_nodes() > 0:
                    print(f"✅ Road network loaded from cache (after wait): {G.number_of_nodes()} nodes")
                    return G
            except Exception:
                pass

    # Download from Overpass as last resort
    data = download_road_network()
    if data is None:
        return None

    # Save to temp cache so other workers can use it
    try:
        with open(TEMP_CACHE, 'w') as f:
            json.dump(data, f)
        print("Saved to temp cache for other workers")
    except Exception:
        pass

    G = build_graph(data)
    if G.number_of_nodes() == 0:
        return None

    print(f"✅ Road network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

print("Loading Pasay road network...")
G = load_graph()
if G is None:
    print("❌ Road network unavailable — /safe-route will return 503")

# ── Node lookup — nearest graph node to a coordinate ─────────────────────────
def nearest_node(lat, lng):
    best_node = None
    best_dist = float('inf')
    for node, data in G.nodes(data=True):
        d = haversine((lat, lng), (data['lat'], data['lng']))
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node

# ── Crime penalty lookup ──────────────────────────────────────────────────────
def get_crime_penalty(lat, lng, heatmap_points):
    if not heatmap_points:
        return 40.0
    best_penalty = 40.0
    best_dist    = float('inf')
    for b in heatmap_points:
        d = haversine((lat, lng), (b['lat'], b['lng']))
        if d < best_dist:
            best_dist    = d
            best_penalty = b['crime_penalty']
    return best_penalty if best_dist < 600 else 40.0

# ── Build crime-weighted graph ────────────────────────────────────────────────
def build_weighted_graph(heatmap_points, lambda_weight=0.5):
    if G is None:
        return None

    penalties = [b['crime_penalty'] for b in heatmap_points] if heatmap_points else [40]
    max_p = max(penalties) or 1
    min_p = min(penalties) or 0
    rng   = (max_p - min_p) or 1

    H = G.copy()
    for u, v, data in H.edges(data=True):
        u_data  = H.nodes[u]
        mid_lat = (u_data['lat'] + H.nodes[v]['lat']) / 2
        mid_lng = (u_data['lng'] + H.nodes[v]['lng']) / 2

        raw_penalty  = get_crime_penalty(mid_lat, mid_lng, heatmap_points)
        norm_penalty = (raw_penalty - min_p) / rng

        base = data.get('travel_time', data.get('length', 1))
        # w'(u,v) = travel_time * (1 + λ * crime_penalty) — thesis formula
        data['safe_weight'] = base * (1 + lambda_weight * norm_penalty)

    return H

# ── Find route ────────────────────────────────────────────────────────────────
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
        path = nx.astar_path(H, orig_node, dest_node,
                             heuristic=lambda u, v: haversine(
                                 (H.nodes[u]['lat'], H.nodes[u]['lng']),
                                 (H.nodes[v]['lat'], H.nodes[v]['lng'])
                             ),
                             weight='safe_weight')
    except nx.NetworkXNoPath:
        path = nx.shortest_path(H, orig_node, dest_node, weight='length')

    polyline = [[H.nodes[n]['lat'], H.nodes[n]['lng']] for n in path]

    total_dist = total_time = total_crime = 0
    for i in range(len(path) - 1):
        u, v   = path[i], path[i+1]
        edge   = H[u][v]
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
        {'id': 'safest',   'label': 'Safest Route',   'tag': '✅ Recommended',
         'desc': 'Avoids high crime-penalty roads.',   'lambda': 1.5},
        {'id': 'balanced', 'label': 'Balanced Route', 'tag': '⚖️ Balanced',
         'desc': 'Moderate crime avoidance.',          'lambda': 0.5},
        {'id': 'fastest',  'label': 'Fastest Route',  'tag': '⚡ Fastest',
         'desc': 'Shortest time, higher crime risk.',  'lambda': 0.0},
    ]

    results = []
    for cfg in configs:
        route = find_route(origin_lat, origin_lng, dest_lat, dest_lng,
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