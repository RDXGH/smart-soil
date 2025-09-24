from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
SOIL_MODEL_PATH = os.path.join(MODEL_DIR, "soil_model.pkl")
SOIL_ENCODER_PATH = os.path.join(MODEL_DIR, "soil_encoder.pkl")

# Load models + encoder
crop_clf = joblib.load(CROP_MODEL_PATH)
soil_clf = joblib.load(SOIL_MODEL_PATH)
soil_encoder = joblib.load(SOIL_ENCODER_PATH)

# Simulated sensor storage
latest_sensor_data = {
    "moisture": 50,
    "temperature": 25,
    "ph": 7.0,
    "npk": 200
}

@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    return jsonify(latest_sensor_data)

@app.route("/api/ingest", methods=["POST"])
def ingest():
    global latest_sensor_data
    data = request.json
    for key in ["moisture", "temperature", "ph", "npk"]:
        if key in data:
            latest_sensor_data[key] = float(data[key])
    return jsonify({"status": "ok", "data": latest_sensor_data})

@app.route("/api/recommendations", methods=["POST"])
def recommend():
    data = request.json
    try:
        # Extract features
        features = [
            float(data["moisture"]),
            float(data["temperature"]),
            float(data["ph"]),
            float(data["npk"])
        ]

        # --- 1) Predict soil type ---
        soil_pred = soil_clf.predict([features])[0]
        encoded_soil = soil_encoder.transform([soil_pred])[0]

        # Add encoded soil type for crop model
        crop_features = features + [encoded_soil]

        # --- 2) Predict crops (top-3) ---
        probs = crop_clf.predict_proba([crop_features])[0]
        top_indices = np.argsort(probs)[::-1][:3]  # top-3
        top_crops = [(crop_clf.classes_[i], probs[i]) for i in top_indices]

        # --- 3) Feature importance analysis ---
        feature_names = ['Moisture', 'Temperature', 'pH', 'NPK', 'Soil Type']
        importances = crop_clf.feature_importances_
        ranked_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [f for f, score in ranked_features[:2]]  # top 2 drivers

        # --- 4) Build recommendations ---
        recommendations = []
        for crop, prob in top_crops:
            recommendations.append({
                "name": crop,
                "probability": f"{prob*100:.1f}%",
                "reason": f"{soil_pred} soil with given conditions favors {crop}. "
                          f"Key factors: {', '.join(top_features)}."
            })

        return jsonify({
            "soil_type": soil_pred,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
