import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# ✅ Use the absolute path you provided
data_path = r"C:\Users\dsunk\Desktop\smart-soil\data\soil_data.csv"
model_out = os.path.join('model', 'crop_model.pkl')

# Load cleaned CSV
df = pd.read_csv(data_path)

# Ensure columns are in expected format
required_cols = ['moisture', 'temperature', 'ph', 'npk', 'crop']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# Select features and target
X = df[['moisture', 'temperature', 'ph', 'npk']]
y = df['crop']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print("✅ Model training complete.")
print("Test accuracy:", clf.score(X_test, y_test))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(clf, model_out)
print(f"✅ Model saved to {model_out}")
