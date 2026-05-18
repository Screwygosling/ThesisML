from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crime_penalty_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except  FileNotFoundError:
    print("Model not found at {MODEL_PATH}. Make sure the .pkl fdile is in the models/folder.")
    model =  None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Crime Threat API is running',
        'endpoints': {
            'POST /predict': 'Send crime features and get a prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
        
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing "features" key in request body.',
                'expected_format': {
                    'features' : [0.5, 1.2, 3.0, '... your feature values here']
                } 
            }), 400
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)

        result = {
            'prediction' : prediction[0].item() if hasattr(prediction[0], 'item')
            else str(prediction[0])
        }

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            result['confidence'] = round(float(np.max(probabilities)), 4)

        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid feature data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)