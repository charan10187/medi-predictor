# train.py  (REPLACE)
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

DATASET_FILE = "Hopsital Dataset.csv"
MODEL_FILE = "model_artifacts.joblib"

def parse_dosage_to_grams(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    # extract a number (with decimal) and unit if present
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(mg|milligram|g|gram|grams)?', s)
    if not m:
        # try to pull any numeric substring
        m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)', s)
        if not m2:
            return np.nan
        val = float(m2.group(1))
        return val  # assume grams if no unit
    val = float(m.group(1))
    unit = m.group(2)
    if unit is None:
        return val  # assume grams
    if unit.startswith('mg'):
        return val / 1000.0
    # grams already
    return val

def normalize_text(v):
    if pd.isna(v): return ""
    return str(v).strip().lower()

def normalize_route(route):
    r = normalize_text(route)
    mapping = {
        'oral': 'Oral',
        'po': 'Oral',
        'intravenous': 'IV',
        'iv': 'IV',
        'im': 'IM',
        'intramuscular': 'IM',
        'subcutaneous': 'SC',
        'sc': 'SC',
        'topical': 'Topical',
        'inhalation': 'Inhalation'
    }
    for k, out in mapping.items():
        if k == r or r.startswith(k):
            return out
    return route if route else "Unknown"

def normalize_frequency(freq):
    f = normalize_text(freq)
    mapping = {
        'once a day': ['once a day','od','1x','1x/day','daily','once/day'],
        'twice a day': ['twice a day','bd','2x','2x/day','bid'],
        'thrice a day': ['tds','3x','3x/day','three times a day'],
        'once weekly': ['once weekly','weekly'],
        'as required': ['prn','as required','when required']
    }
    for out, options in mapping.items():
        for opt in options:
            if opt == f or f.startswith(opt) or opt in f:
                return out.title()
    # fallback: capitalize if short
    return freq.title() if freq else "Unknown"

def train_all_models():
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"Loaded {len(df)} rows from '{DATASET_FILE}'")
    except FileNotFoundError:
        print(f"Dataset not found: {DATASET_FILE}")
        return

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    required = ['Age', 'Gender', 'Diagnosis', 'Name of Drug', 'Dosage (gram)', 'Route', 'Frequency']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        return

    # Age -> numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Dosage: robust parse to grams from text
    df['Dosage_parsed'] = df['Dosage (gram)'].apply(parse_dosage_to_grams)

    # Normalize text columns
    df['Gender'] = df['Gender'].astype(str).str.strip()
    df['Diagnosis'] = df['Diagnosis'].astype(str).str.strip()
    df['Route'] = df['Route'].apply(normalize_route)
    df['Frequency'] = df['Frequency'].apply(normalize_frequency)
    df['Name of Drug'] = df['Name of Drug'].astype(str).str.strip()

    # Drop rows missing essential outputs or age
    df = df.dropna(subset=['Age', 'Gender', 'Diagnosis', 'Name of Drug'])
    # For dosage, fill NaN with median of parsed
    median_dosage = df['Dosage_parsed'].median(skipna=True)
    df['Dosage_parsed'] = df['Dosage_parsed'].fillna(median_dosage if not np.isnan(median_dosage) else 0.0)

    # Convert Age to int
    df['Age'] = df['Age'].astype(int)

    # Prepare inputs and outputs
    input_features = ['Age', 'Gender', 'Diagnosis']
    output_targets = ['Name of Drug', 'Dosage_parsed', 'Route', 'Frequency']

    X = df[input_features].copy()
    y = df[['Name of Drug', 'Dosage_parsed', 'Route', 'Frequency']].copy()

    # Build simple mapping dictionaries for categorical inputs
    # Use most_common fallback values
    def build_map(series):
        vals = [v for v in series.astype(str).tolist() if v and v.lower() != 'nan']
        cnt = Counter(vals)
        uniques = sorted(list(dict.fromkeys(vals)))  # keep insertion order uniqueness
        most_common = cnt.most_common(1)[0][0] if cnt else (uniques[0] if uniques else "")
        mapping = {v: i for i, v in enumerate(uniques)}
        return mapping, most_common

    gender_map, gender_default = build_map(X['Gender'])
    diagnosis_map, diagnosis_default = build_map(X['Diagnosis'])

    # Apply maps (map unseen to default index)
    X['Gender_enc'] = X['Gender'].map(gender_map).fillna(gender_map.get(gender_default, 0)).astype(int)
    X['Diagnosis_enc'] = X['Diagnosis'].map(diagnosis_map).fillna(diagnosis_map.get(diagnosis_default, 0)).astype(int)

    X_model = X[['Age', 'Gender_enc', 'Diagnosis_enc']].copy()

    # Train targets
    label_encoders = {}
    models = {}

    # Drug classifier
    le_drug = LabelEncoder()
    y_drug = le_drug.fit_transform(y['Name of Drug'])
    label_encoders['Name of Drug'] = le_drug
    clf_drug = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_drug.fit(X_model, y_drug)
    models['drug'] = clf_drug

    # Dosage regressor
    reg_dosage = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_dosage.fit(X_model, y['Dosage_parsed'])
    models['dosage'] = reg_dosage

    # Route classifier
    le_route = LabelEncoder()
    y_route = le_route.fit_transform(y['Route'])
    label_encoders['Route'] = le_route
    clf_route = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_route.fit(X_model, y_route)
    models['route'] = clf_route

    # Frequency classifier
    le_freq = LabelEncoder()
    y_freq = le_freq.fit_transform(y['Frequency'])
    label_encoders['Frequency'] = le_freq
    clf_freq = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_freq.fit(X_model, y_freq)
    models['frequency'] = clf_freq

    # Save artifacts including mappings and defaults and diagnoses list
    artifacts = {
        'models': models,
        'label_encoders': label_encoders,
        'input_features': input_features,
        'input_mappings': {
            'gender_map': gender_map,
            'gender_default': gender_default,
            'diagnosis_map': diagnosis_map,
            'diagnosis_default': diagnosis_default
        },
        'diagnoses': sorted(list(diagnosis_map.keys()))
    }
    joblib.dump(artifacts, MODEL_FILE)
    print(f"Saved artifacts to {MODEL_FILE}")

if __name__ == '__main__':
    train_all_models()
