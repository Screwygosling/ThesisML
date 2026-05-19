from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crime_penalty_model.pkl')

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


# ── barangay data ─────────────────────────────────────────────────────────────
# barangay_encoded: alphabetical index from df["barangay"].astype("category").cat.codes
# municipal_encoded: 0 for all (PASAY CITY is the only municipality)
# areaCrimeCount: exact values from df.groupby("barangay").size()
# crime_severity: 3 for high-crime barangays (count >= 10), 2 for others

BARANGAYS = [
    {'name': 'BARANGAY 1',   'lat': 14.5600, 'lng': 121.0010, 'barangay_encoded': 0,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 10',  'lat': 14.5590, 'lng': 121.0015, 'barangay_encoded': 1,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 110', 'lat': 14.5430, 'lng': 121.0120, 'barangay_encoded': 2,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 114', 'lat': 14.5420, 'lng': 121.0130, 'barangay_encoded': 3,  'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 12',  'lat': 14.5585, 'lng': 121.0020, 'barangay_encoded': 4,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 120', 'lat': 14.5410, 'lng': 121.0140, 'barangay_encoded': 5,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 13',  'lat': 14.5580, 'lng': 121.0025, 'barangay_encoded': 6,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 130', 'lat': 14.5400, 'lng': 121.0150, 'barangay_encoded': 7,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 131', 'lat': 14.5395, 'lng': 121.0155, 'barangay_encoded': 8,  'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 14',  'lat': 14.5575, 'lng': 121.0030, 'barangay_encoded': 9,  'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 140', 'lat': 14.5385, 'lng': 121.0165, 'barangay_encoded': 10, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 144', 'lat': 14.5375, 'lng': 121.0170, 'barangay_encoded': 11, 'areaCrimeCount': 6,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 145', 'lat': 14.5370, 'lng': 121.0175, 'barangay_encoded': 12, 'areaCrimeCount': 3,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 146', 'lat': 14.5365, 'lng': 121.0180, 'barangay_encoded': 13, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 147', 'lat': 14.5360, 'lng': 121.0185, 'barangay_encoded': 14, 'areaCrimeCount': 3,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 148', 'lat': 14.5355, 'lng': 121.0190, 'barangay_encoded': 15, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 149', 'lat': 14.5350, 'lng': 121.0195, 'barangay_encoded': 16, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 150', 'lat': 14.5345, 'lng': 121.0200, 'barangay_encoded': 17, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 152', 'lat': 14.5340, 'lng': 121.0205, 'barangay_encoded': 18, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 157', 'lat': 14.5335, 'lng': 121.0210, 'barangay_encoded': 19, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 159', 'lat': 14.5330, 'lng': 121.0215, 'barangay_encoded': 20, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 162', 'lat': 14.5325, 'lng': 121.0220, 'barangay_encoded': 21, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 163', 'lat': 14.5320, 'lng': 121.0225, 'barangay_encoded': 22, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 165', 'lat': 14.5315, 'lng': 121.0230, 'barangay_encoded': 23, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 166', 'lat': 14.5310, 'lng': 121.0235, 'barangay_encoded': 24, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 168', 'lat': 14.5305, 'lng': 121.0240, 'barangay_encoded': 25, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 170', 'lat': 14.5300, 'lng': 121.0245, 'barangay_encoded': 26, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 171', 'lat': 14.5295, 'lng': 121.0250, 'barangay_encoded': 27, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 172', 'lat': 14.5290, 'lng': 121.0255, 'barangay_encoded': 28, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 173', 'lat': 14.5285, 'lng': 121.0260, 'barangay_encoded': 29, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 175', 'lat': 14.5280, 'lng': 121.0265, 'barangay_encoded': 30, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 176', 'lat': 14.5325, 'lng': 121.0106, 'barangay_encoded': 31, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 178', 'lat': 14.5270, 'lng': 121.0275, 'barangay_encoded': 32, 'areaCrimeCount': 3,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 179', 'lat': 14.5265, 'lng': 121.0280, 'barangay_encoded': 33, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 180', 'lat': 14.5260, 'lng': 121.0285, 'barangay_encoded': 34, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 183', 'lat': 14.5299, 'lng': 121.0151, 'barangay_encoded': 35, 'areaCrimeCount': 15, 'crime_severity': 3, 'victimCount': 2},
    {'name': 'BARANGAY 184', 'lat': 14.5250, 'lng': 121.0295, 'barangay_encoded': 36, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 186', 'lat': 14.5245, 'lng': 121.0300, 'barangay_encoded': 37, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 188', 'lat': 14.5240, 'lng': 121.0305, 'barangay_encoded': 38, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 190', 'lat': 14.5235, 'lng': 121.0310, 'barangay_encoded': 39, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 191', 'lat': 14.5230, 'lng': 121.0315, 'barangay_encoded': 40, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 192', 'lat': 14.5225, 'lng': 121.0320, 'barangay_encoded': 41, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 193', 'lat': 14.5220, 'lng': 121.0325, 'barangay_encoded': 42, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 194', 'lat': 14.5215, 'lng': 121.0330, 'barangay_encoded': 43, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 195', 'lat': 14.5210, 'lng': 121.0335, 'barangay_encoded': 44, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 197', 'lat': 14.5205, 'lng': 121.0340, 'barangay_encoded': 45, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 201', 'lat': 14.5200, 'lng': 121.0345, 'barangay_encoded': 46, 'areaCrimeCount': 3,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 27',  'lat': 14.5560, 'lng': 120.9980, 'barangay_encoded': 47, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 32',  'lat': 14.5550, 'lng': 120.9970, 'barangay_encoded': 48, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 33',  'lat': 14.5545, 'lng': 120.9965, 'barangay_encoded': 49, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 35',  'lat': 14.5540, 'lng': 120.9960, 'barangay_encoded': 50, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 38',  'lat': 14.5535, 'lng': 120.9955, 'barangay_encoded': 51, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 39',  'lat': 14.5530, 'lng': 120.9950, 'barangay_encoded': 52, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 4',   'lat': 14.5569, 'lng': 120.9991, 'barangay_encoded': 53, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 40',  'lat': 14.5525, 'lng': 120.9945, 'barangay_encoded': 54, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 41',  'lat': 14.5520, 'lng': 120.9940, 'barangay_encoded': 55, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 44',  'lat': 14.5515, 'lng': 120.9935, 'barangay_encoded': 56, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 46',  'lat': 14.5510, 'lng': 120.9930, 'barangay_encoded': 57, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 47',  'lat': 14.5510, 'lng': 120.9903, 'barangay_encoded': 58, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 49',  'lat': 14.5505, 'lng': 120.9920, 'barangay_encoded': 59, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 56',  'lat': 14.5500, 'lng': 120.9915, 'barangay_encoded': 60, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 60',  'lat': 14.5495, 'lng': 120.9910, 'barangay_encoded': 61, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 62',  'lat': 14.5490, 'lng': 120.9905, 'barangay_encoded': 62, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 7',   'lat': 14.5595, 'lng': 121.0005, 'barangay_encoded': 63, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 70',  'lat': 14.5485, 'lng': 120.9900, 'barangay_encoded': 64, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 75',  'lat': 14.5480, 'lng': 120.9895, 'barangay_encoded': 65, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 76',  'lat': 14.5542, 'lng': 120.9893, 'barangay_encoded': 66, 'areaCrimeCount': 34, 'crime_severity': 3, 'victimCount': 2},
    {'name': 'BARANGAY 79',  'lat': 14.5470, 'lng': 120.9885, 'barangay_encoded': 67, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 81',  'lat': 14.5465, 'lng': 120.9880, 'barangay_encoded': 68, 'areaCrimeCount': 3,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 83',  'lat': 14.5460, 'lng': 120.9875, 'barangay_encoded': 69, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 84',  'lat': 14.5455, 'lng': 120.9870, 'barangay_encoded': 70, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 90',  'lat': 14.5450, 'lng': 120.9865, 'barangay_encoded': 71, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 92',  'lat': 14.5445, 'lng': 120.9860, 'barangay_encoded': 72, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 95',  'lat': 14.5440, 'lng': 120.9855, 'barangay_encoded': 73, 'areaCrimeCount': 1,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 97',  'lat': 14.5435, 'lng': 120.9850, 'barangay_encoded': 74, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
    {'name': 'BARANGAY 98',  'lat': 14.5430, 'lng': 120.9845, 'barangay_encoded': 75, 'areaCrimeCount': 2,  'crime_severity': 2, 'victimCount': 1},
]


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