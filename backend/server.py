import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)
CORS(app)

# SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sensors.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class SensorReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    moisture = db.Column(db.Float, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    ph = db.Column(db.Float, nullable=True)
    npk = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'moisture': self.moisture,
            'temperature': self.temperature,
            'ph': self.ph,
            'npk': self.npk
        }

with app.app_context():
    db.create_all()

# ==============================
# Load CSV + Train Model
# ==============================
MODEL_PATH = os.path.join('model', 'crop_model.pkl')
DATASET_PATH = os.path.join('model', 'dataset.csv')

ml_model = None
if os.path.exists(DATASET_PATH):
    try:
        df = pd.read_csv(DATASET_PATH)
        X = df[['moisture', 'temperature', 'ph', 'npk']]
        y = df['crop']
        ml_model = DecisionTreeClassifier()
        ml_model.fit(X, y)

        # Save model for future use
        os.makedirs("model", exist_ok=True)
        joblib.dump(ml_model, MODEL_PATH)
        print("✅ Model trained from CSV and saved.")
    except Exception as e:
        print("❌ Error training model from CSV:", e)
else:
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            print("✅ ML model loaded from file.")
        except:
            print("❌ No dataset.csv or valid model found. Cannot predict.")

# ==============================
# API Routes
# ==============================

@app.route('/api/ingest', methods=['POST'])
def ingest():
    data = request.get_json() or {}
    try:
        reading = SensorReading(
            moisture=float(data.get('moisture')) if data.get('moisture') else None,
            temperature=float(data.get('temperature')) if data.get('temperature') else None,
            ph=float(data.get('ph')) if data.get('ph') else None,
            npk=float(data.get('npk')) if data.get('npk') else None
        )
        db.session.add(reading)
        db.session.commit()
        return jsonify({'status': 'ok', 'id': reading.id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sensors', methods=['GET'])
def get_latest_sensor():
    r = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    if not r:
        return jsonify({'moisture': 0, 'temperature': 0, 'ph': 7.0, 'npk': 120})
    return jsonify(r.to_dict())


# ✅ Fixed: POST method for recommendations
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json() or {}

    # Fallback: use latest DB reading if no data provided
    if not data:
        r = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
        if r:
            data = r.to_dict()
        else:
            data = {'moisture': 0, 'temperature': 0, 'ph': 7.0, 'npk': 120}

    if ml_model:
        try:
            X = [[
                float(data.get('moisture', 0)),
                float(data.get('temperature', 0)),
                float(data.get('ph', 7.0)),
                float(data.get('npk', 120))
            ]]
            pred = ml_model.predict(X)[0]
            return jsonify({
                'recommendations': [
                    {'name': pred, 'reason': 'Predicted from trained crop model'}
                ]
            })
        except Exception as e:
            return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500
    else:
        return jsonify({'error': 'No ML model available. Please retrain.'}), 500


@app.route('/api/readings', methods=['GET'])
def get_readings():
    n = int(request.args.get('n', 50))
    rows = SensorReading.query.order_by(SensorReading.timestamp.desc()).limit(n).all()
    return jsonify([r.to_dict() for r in rows])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
