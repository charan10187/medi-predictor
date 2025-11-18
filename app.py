# app.py (REPLACE)
import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

MODEL_FILE = "model_artifacts.joblib"

if not os.path.exists(MODEL_FILE):
    raise SystemExit(f"Model artifacts not found at '{MODEL_FILE}'. Run train.py first.")

artifacts = joblib.load(MODEL_FILE)
models = artifacts['models']
label_encoders = artifacts['label_encoders']
input_features = artifacts['input_features']
input_mappings = artifacts.get('input_mappings', {})
diagnoses_for_dropdown = artifacts.get('diagnoses', [])

print("Loaded model artifacts.")

def encode_input(data):
    # Expect keys: 'Age', 'Gender', 'Diagnosis'
    age = data.get('Age', None)
    try:
        age = int(float(age))
    except Exception:
        age = 0

    gender = str(data.get('Gender', '')).strip()
    diagnosis = str(data.get('Diagnosis', '')).strip()

    gm = input_mappings.get('gender_map', {})
    gd = input_mappings.get('gender_default', next(iter(gm), None))
    dm = input_mappings.get('diagnosis_map', {})
    dd = input_mappings.get('diagnosis_default', next(iter(dm), None))

    gender_enc = gm.get(gender, gm.get(gd, 0))
    diagnosis_enc = dm.get(diagnosis, dm.get(dd, 0))

    return pd.DataFrame([[age, int(gender_enc), int(diagnosis_enc)]], columns=['Age', 'Gender_enc', 'Diagnosis_enc'])

@app.route('/')
def index():
    return render_template('index.html', diagnoses=diagnoses_for_dropdown)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_in = encode_input(data)

        # Ensure order matches training: Age, Gender_enc, Diagnosis_enc
        X = df_in[['Age', 'Gender_enc', 'Diagnosis_enc']]

        pred_drug_encoded = models['drug'].predict(X)
        pred_dosage = models['dosage'].predict(X)
        pred_route_encoded = models['route'].predict(X)
        pred_freq_encoded = models['frequency'].predict(X)

        pred_drug = label_encoders['Name of Drug'].inverse_transform(pred_drug_encoded)[0]
        pred_route = label_encoders['Route'].inverse_transform(pred_route_encoded)[0]
        pred_freq = label_encoders['Frequency'].inverse_transform(pred_freq_encoded)[0]

        response = {
            'drug': pred_drug,
            'dosage': f"{pred_dosage[0]:.3f} g",
            'route': pred_route,
            'frequency': pred_freq
        }
        return jsonify(response)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
