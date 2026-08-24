import requests, json

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

resp = requests.post(
    'https://overpass-api.de/api/interpreter',
    data={'data': query},
    headers={'User-Agent': 'SafeCommutePH/1.0 (thesis research project)'},
    timeout=60
)
with open('pasay_roads.json', 'w') as f:
    json.dump(resp.json(), f)
print('Done:', len(resp.json()['elements']), 'elements')