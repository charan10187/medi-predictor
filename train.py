import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
# IMPORTANT: Change this to the exact name of your Google Sheet
GOOGLE_SHEET_NAME = "Your Google Sheet Name Here"
WORKSHEET_NAME = "data"
CREDENTIALS_FILE = "credentials.json"
MODEL_FILE = "model_artifacts.joblib" # Saves the model in your project folder

def get_data_from_google_sheet():
    """Fetches data from Google Sheet and returns a Pandas DataFrame."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        data = sheet.get_all_records()
        print("✅ Data loaded successfully from Google Sheet.")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def train_model():
    """Trains the LightGBM model and saves the artifacts."""
    df = get_data_from_google_sheet()
    if df is None:
        return

    # --- Data Cleaning ---
    if 'Date of Data Entry' in df.columns:
        df = df.drop(columns=['Date of Data Entry'])
    df.dropna(inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # --- Preprocessing ---
    target_column = 'Name of Drug'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    label_encoders[target_column] = le_target
    
    feature_columns = X.columns.tolist()

    # --- Model Training ---
    print("🚀 Starting model training with LightGBM...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X, y_encoded)
    print("✅ Model training complete.")

    # --- Save Artifacts ---
    artifacts = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }
    joblib.dump(artifacts, MODEL_FILE)
    print(f"💾 Model artifacts saved to '{MODEL_FILE}'")

if __name__ == '__main__':
    train_model()
