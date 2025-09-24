import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt

# ✅ Path to your dataset
data_path = r"C:\Users\dsunk\Desktop\smart-soil\data\soil_data.csv"
model_out_crop = os.path.join('model', 'crop_model.pkl')
model_out_soil = os.path.join('model', 'soil_model.pkl')
encoder_out = os.path.join('model', 'soil_encoder.pkl')

# Load dataset
df = pd.read_csv(data_path)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# ✅ Encode soil type
if 'soil_type' not in df.columns:
    raise ValueError("Dataset must have a 'soil_type' column")

soil_encoder = LabelEncoder()
df['soil_type_encoded'] = soil_encoder.fit_transform(df['soil_type'])

# Save encoder
os.makedirs('model', exist_ok=True)
joblib.dump(soil_encoder, encoder_out)

# Features & targets
X = df[['moisture', 'temperature', 'ph', 'npk']]
y_soil = df['soil_type']
y_crop = df['crop']

# --- Train soil type model ---
X_train, X_test, y_train, y_test = train_test_split(X, y_soil, test_size=0.2, random_state=42)
soil_clf = RandomForestClassifier(n_estimators=200, random_state=42)
soil_clf.fit(X_train, y_train)
print("✅ Soil type model accuracy:", soil_clf.score(X_test, y_test))

# Save soil model
joblib.dump(soil_clf, model_out_soil)

# --- Train crop model (uses soil_type_encoded too) ---
X_crop = df[['moisture', 'temperature', 'ph', 'npk', 'soil_type_encoded']]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

crop_clf = RandomForestClassifier(n_estimators=300, random_state=42)
crop_clf.fit(Xc_train, yc_train)
print("✅ Crop model accuracy:", crop_clf.score(Xc_test, yc_test))

# Save crop model
joblib.dump(crop_clf, model_out_crop)

# --- Feature Importance Analysis ---
def plot_feature_importance(model, feature_names, title, file_name):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    fi.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    out_path = os.path.join("model", file_name)
    plt.savefig(out_path)
    print(f"📊 Saved feature importance chart: {out_path}")

# Soil model feature importance
plot_feature_importance(
    soil_clf,
    ['Moisture', 'Temperature', 'pH', 'NPK'],
    "Soil Type Prediction - Feature Importance",
    "soil_feature_importance.png"
)

# Crop model feature importance
plot_feature_importance(
    crop_clf,
    ['Moisture', 'Temperature', 'pH', 'NPK', 'Soil Type Encoded'],
    "Crop Prediction - Feature Importance",
    "crop_feature_importance.png"
)

print("✅ Models trained & saved successfully.")
