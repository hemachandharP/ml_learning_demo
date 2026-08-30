from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Diabetes prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        required_fields = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]

        # Check for missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Keep EXACTLY the same order used during training
        features = [
            data["gender"],
            data["age"],
            data["hypertension"],
            data["heart_disease"],
            data["smoking_history"],
            data["bmi"],
            data["HbA1c_level"],
            data["blood_glucose_level"]
        ]

        # Convert to NumPy array
        features_array = np.array(
            [features],
            dtype=float
        )

        # Scale the input
        scaled_features = scaler.transform(features_array)

        # Prediction
        prediction = model.predict(scaled_features)[0]

        # Probability
        probability = model.predict_proba(
            scaled_features
        )[0][1]

        return jsonify({
            "diabetes_prediction": int(prediction),
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
