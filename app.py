from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

model = None
label_encoder = None
feature_columns = None
categorical_cols = [
    'Gender',
    'Diagnosis',
    'Antibiotic_Resistance_Test',
    'Complications',
    'Comorbidities'
]

def train_and_load_model():
    global model, label_encoder, feature_columns

    try:
        df = pd.read_csv('pulmonology_treatment_dataset.csv')
        print("Dataset loaded successfully for backend training.")
    except FileNotFoundError:
        print("Error: 'pulmonology_treatment_dataset.csv' not found.")
        return False
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    features_df = df.drop(['Treatment_Outcome'], axis=1)
    features_processed = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

    X = features_processed.drop('Antibiotic_Treatment', axis=1)
    y = features_processed['Antibiotic_Treatment']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("Backend: Performing Hyperparameter Tuning with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=3,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=0)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Backend: Best parameters found: {best_params}")

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    print("Backend: Model training complete.")
    return True

# Train model on startup
if not train_and_load_model():
    print("Backend: Model training failed. Exiting.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_antibiotic():
    if model is None or label_encoder is None or feature_columns is None:
        return jsonify({'error': 'Model not loaded. Please check backend logs.'}), 500

    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        input_df = pd.DataFrame([data])

        for col in ['Complications', 'Comorbidities']:
            if col in input_df.columns:
                input_df[col] = input_df[col].fillna('None')

        processed_input = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        missing_cols = set(feature_columns) - set(processed_input.columns)
        for c in missing_cols:
            processed_input[c] = 0

        processed_input = processed_input[feature_columns]

        print("Processed input for prediction:")
        print(processed_input)

        prediction_encoded = model.predict(processed_input)
        predicted_antibiotic = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({'prediction': predicted_antibiotic})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
