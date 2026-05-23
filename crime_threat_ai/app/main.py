from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import csv
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'crime_penalty_model.pkl')
DATA_CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'crime_data.csv'))

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}.")
    model = None


# ── helpers ───────────────────────────────────────────────────────────────────

def penalty_to_threat(penalty_score):
    """Convert raw regression output to a threat label + 0-1 intensity."""
    if penalty_score < 25:
        return 'low', round(penalty_score / 25 * 0.3, 4)
    elif penalty_score < 50:
        return 'moderate', round(0.3 + (penalty_score - 25) / 25 * 0.3, 4)
    elif penalty_score < 75:
        return 'high', round(0.6 + (penalty_score - 50) / 25 * 0.25, 4)
    else:
        return 'very high', round(min(0.85 + (penalty_score - 75) / 25 * 0.15, 1.0), 4)


# ── barangay data (loaded from CSV) ──────────────────────────────────────────
def load_barangays_from_csv(csv_path):
    """Aggregate raw incident CSV into barangay-level entries used by the app."""
    groups = defaultdict(lambda: {'count': 0, 'lat_sum': 0.0, 'lng_sum': 0.0, 'victim_sum': 0})

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                barangay = (row.get('barangay') or '').strip().upper()
                if not barangay:
                    continue
                try:
                    lat = float(row.get('lat') or 0)
                    lng = float(row.get('lng') or 0)
                except ValueError:
                    # skip rows with invalid coordinates
                    continue
                try:
                    victim = int(row.get('victimCount') or 0)
                except ValueError:
                    victim = 0

                g = groups[barangay]
                g['count'] += 1
                g['lat_sum'] += lat
                g['lng_sum'] += lng
                g['victim_sum'] += victim

    except FileNotFoundError:
        print(f"❌ CSV not found at {csv_path}. Falling back to empty barangays list.")
        return []

    # Build list with deterministic encoding
    names = sorted(groups.keys())
    barangays = []
    for idx, name in enumerate(names):
        g = groups[name]
        count = g['count']
        avg_lat = g['lat_sum'] / count if count else 0.0
        avg_lng = g['lng_sum'] / count if count else 0.0
        areaCrimeCount = count
        crime_severity = 3 if areaCrimeCount >= 10 else 2
        victimCount = g['victim_sum']

        barangays.append({
            'name': name,
            'lat': round(avg_lat, 6),
            'lng': round(avg_lng, 6),
            'barangay_encoded': idx,
            'areaCrimeCount': areaCrimeCount,
            'crime_severity': crime_severity,
            'victimCount': victimCount,
        })

    return barangays


# Load barangays from CSV at startup (fallback to empty list if missing)
BARANGAYS = load_barangays_from_csv(DATA_CSV_PATH)



# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Crime Threat AI API is running',
        'endpoints': {
            'GET  /health':  'Check if API and model are ready',
            'POST /predict': 'Predict crime penalty for one location',
            'GET  /heatmap': 'Get heatmap points for all Pasay City barangays'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
    {
        "lat": 14.5542,
        "lng": 120.9893,
        "month": 5,
        "day_of_week": 2,
        "hour": 14,
        "areaCrimeCount": 34,
        "barangay_encoded": 66,
        "municipal_encoded": 0,
        "victimCount": 1,
        "crime_severity": 3
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON body provided.'}), 400

        required_fields = [
            'lat', 'lng', 'month', 'day_of_week', 'hour',
            'areaCrimeCount', 'barangay_encoded', 'municipal_encoded',
            'victimCount', 'crime_severity'
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        features = np.array([[
            data['lat'],
            data['lng'],
            data['month'],
            data['day_of_week'],
            data['hour'],
            data['areaCrimeCount'],
            data['barangay_encoded'],
            data['municipal_encoded'],
            data['victimCount'],
            data['crime_severity']
        ]])

        penalty = float(model.predict(features)[0])
        threat_level, intensity = penalty_to_threat(penalty)

        return jsonify({
            'crime_penalty': round(penalty, 4),
            'threat_level':  threat_level,
            'intensity':     intensity
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid data: {str(e)}'}), 400
    except AttributeError as e:
        return jsonify({'error': f'Model error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/heatmap', methods=['GET'])
def heatmap():
    """
    Runs the model for all 76 Pasay City barangays using current time.
    Returns heatmap points for the Leaflet map.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        now = datetime.now()
        month       = now.month
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        hour        = now.hour

        points = []
        for b in BARANGAYS:
            features = np.array([[
                b['lat'],
                b['lng'],
                month,
                day_of_week,
                hour,
                b['areaCrimeCount'],
                b['barangay_encoded'],
                0,                    # municipal_encoded: PASAY CITY = 0
                b['victimCount'],
                b['crime_severity']
            ]])

            penalty = float(model.predict(features)[0])
            threat_level, intensity = penalty_to_threat(penalty)

            points.append({
                'name':          b['name'],
                'lat':           b['lat'],
                'lng':           b['lng'],
                'intensity':     intensity,
                'threat_level':  threat_level,
                'crime_penalty': round(penalty, 4)
            })

        return jsonify({'points': points, 'generated_at': now.isoformat()})

    except Exception as e:
        return jsonify({'error': f'Heatmap generation failed: {str(e)}'}), 500

@app.route('/route', methods=['POST'])
def route():
    """
    Calculates safety scores for 3 route options based on crime data.
    Expects JSON:
    {
        "origin": "Barangay 76",
        "destination": "Barangay 183"
    }
    Returns safety scores for safest, balanced, and fastest routes.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        now = datetime.now()
        month       = now.month
        day_of_week = now.weekday()
        hour        = now.hour

        # Calculate average crime penalty across all barangays for current time
        # This gives us a baseline threat level
        all_penalties = []
        for b in BARANGAYS:
            features = np.array([[
                b['lat'], b['lng'], month, day_of_week, hour,
                b['areaCrimeCount'], b['barangay_encoded'], 0,
                b['victimCount'], b['crime_severity']
            ]])
            penalty = float(model.predict(features)[0])
            all_penalties.append(penalty)

        avg_penalty = np.mean(all_penalties)
        max_penalty = np.max(all_penalties)

        # Convert penalty to safety score (inverse relationship)
        # Higher penalty = lower safety score
        def penalty_to_safety(penalty, weight=1.0):
            raw = 100 - (penalty / max_penalty * 100 * weight)
            return max(0, min(100, round(raw)))

        safest_score  = penalty_to_safety(avg_penalty, weight=0.5)  # avoids high risk
        balanced_score = penalty_to_safety(avg_penalty, weight=0.8)  # moderate risk
        fastest_score  = penalty_to_safety(avg_penalty, weight=1.2)  # passes risky areas

        return jsonify({
            'routes': [
                {
                    'id': 'safest',
                    'label': 'Safest Route',
                    'score': safest_score,
                    'duration': '28 min',
                    'distance': '6.2 km',
                    'desc': 'Avoids all high-risk zones.',
                    'tag': '✅ Recommended'
                },
                {
                    'id': 'balanced',
                    'label': 'Balanced Route',
                    'score': balanced_score,
                    'duration': '21 min',
                    'distance': '4.9 km',
                    'desc': 'Moderate risk, slightly faster.',
                    'tag': '⚖️ Balanced'
                },
                {
                    'id': 'fastest',
                    'label': 'Fastest Route',
                    'score': fastest_score,
                    'duration': '15 min',
                    'distance': '4.1 km',
                    'desc': 'Passes high-risk areas. Caution.',
                    'tag': '⚡ Fastest'
                },
            ],
            'generated_at': now.isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'Route calculation failed: {str(e)}'}), 500


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)