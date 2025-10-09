import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DATASET_FILE = "Hopsital Dataset.csv"
MODEL_FILE = "model_artifacts.joblib"

def train_all_models():
    """
    Trains four separate models for Drug, Dosage, Route, and Frequency
    and saves them all as a single artifact.
    """
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"✅ Data loaded successfully from '{DATASET_FILE}'.")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{DATASET_FILE}'.")
        return

    # --- 1. Data Cleaning & Preparation ---

    # --- NEW: Clean the Age column ---
    # Convert 'Age' to a numeric type. If any value can't be converted (like the word 'Age'),
    # it will be replaced with NaN (Not a Number).
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Drop any rows where 'Age' is now NaN, and also drop rows with missing essential data.
    df.dropna(subset=['Age', 'Gender', 'Diagnosis', 'Name of Drug', 'Dosage (gram)', 'Route', 'Frequency'], inplace=True)

    # Ensure 'Age' is an integer after cleaning
    df['Age'] = df['Age'].astype(int)
    
    # Define inputs and outputs
    input_features = ['Age', 'Gender', 'Diagnosis']
    output_targets = ['Name of Drug', 'Dosage (gram)', 'Route', 'Frequency']
    
    X = df[input_features].copy() # Use .copy() to avoid SettingWithCopyWarning
    y = df[output_targets]

    # --- 2. Preprocessing ---
    label_encoders = {}
    
    # Encode categorical input features
    for col in ['Gender', 'Diagnosis']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # --- 3. Train a Model for Each Target ---
    models = {}
    
    # --- Target 1: Name of Drug (Classifier) ---
    print("🚀 Training Drug Name model...")
    le_drug = LabelEncoder()
    y_drug_encoded = le_drug.fit_transform(y['Name of Drug'])
    label_encoders['Name of Drug'] = le_drug
    model_drug = RandomForestClassifier(n_estimators=100, random_state=42)
    model_drug.fit(X, y_drug_encoded)
    models['drug'] = model_drug
    print("✅ Drug Name model trained.")

    # --- Target 2: Dosage (Regressor) ---
    print("🚀 Training Dosage model...")
    y_dosage = y['Dosage (gram)']
    y_dosage = pd.to_numeric(y_dosage, errors='coerce').fillna(0)
    model_dosage = RandomForestRegressor(n_estimators=100, random_state=42)
    model_dosage.fit(X, y_dosage)
    models['dosage'] = model_dosage
    print("✅ Dosage model trained.")

    # --- Target 3: Route (Classifier) ---
    print("🚀 Training Route model...")
    le_route = LabelEncoder()
    y_route_encoded = le_route.fit_transform(y['Route'])
    label_encoders['Route'] = le_route
    model_route = RandomForestClassifier(n_estimators=100, random_state=42)
    model_route.fit(X, y_route_encoded)
    models['route'] = model_route
    print("✅ Route model trained.")

    # --- Target 4: Frequency (Classifier) ---
    print("🚀 Training Frequency model...")
    le_freq = LabelEncoder()
    y_freq_encoded = le_freq.fit_transform(y['Frequency'])
    label_encoders['Frequency'] = le_freq
    model_freq = RandomForestClassifier(n_estimators=100, random_state=42)
    model_freq.fit(X, y_freq_encoded)
    models['frequency'] = model_freq
    print("✅ Frequency model trained.")

    # --- 4. Save All Artifacts ---
    artifacts = {
        'models': models,
        'label_encoders': label_encoders,
        'input_features': input_features
    }
    joblib.dump(artifacts, MODEL_FILE)
    print(f"\n💾 All models and artifacts saved to '{MODEL_FILE}'")

if __name__ == '__main__':
    train_all_models()