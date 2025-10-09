import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# --- Load All Model Artifacts at Startup ---
MODEL_FILE = "model_artifacts.joblib"

if not os.path.exists(MODEL_FILE):
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_FILE}'!")
    print("Please run 'train.py' first to create the model artifacts.")
    exit()

artifacts = joblib.load(MODEL_FILE)
models = artifacts['models']
label_encoders = artifacts['label_encoders']
input_features = artifacts['input_features']
print(f"✅ All model artifacts loaded successfully from '{MODEL_FILE}'.")


@app.route('/')
def index():
    # Load the dataset to get unique diagnoses for the dropdown
    try:
        df = pd.read_csv("Hopsital Dataset.csv")
        # Get unique, sorted, non-empty diagnosis values
        diagnoses = sorted([str(d) for d in df['Diagnosis'].dropna().unique() if d])
    except FileNotFoundError:
        diagnoses = [] # In case the file is not found, provide an empty list
        print("⚠️ Warning: 'Hopsital Dataset.csv' not found. Diagnosis dropdown will be empty.")

    return render_template('index.html', diagnoses=diagnoses)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Create a DataFrame from the input
        input_df = pd.DataFrame([data], columns=input_features)

        # Preprocess input data using loaded encoders
        for col in ['Gender', 'Diagnosis']:
            le = label_encoders[col]
            # Handle unseen labels by mapping to the first known class
            input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])
        
        # Ensure all columns are numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # --- Make Predictions with Each Model ---
        pred_drug_encoded = models['drug'].predict(input_df)
        pred_dosage = models['dosage'].predict(input_df)
        pred_route_encoded = models['route'].predict(input_df)
        pred_freq_encoded = models['frequency'].predict(input_df)

        # --- Decode Predictions ---
        pred_drug = label_encoders['Name of Drug'].inverse_transform(pred_drug_encoded)[0]
        pred_route = label_encoders['Route'].inverse_transform(pred_route_encoded)[0]
        pred_freq = label_encoders['Frequency'].inverse_transform(pred_freq_encoded)[0]

        # Format the response
        response = {
            'drug': pred_drug,
            'dosage': f"{pred_dosage[0]:.2f} grams", # Format dosage to 2 decimal places
            'route': pred_route,
            'frequency': pred_freq
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)