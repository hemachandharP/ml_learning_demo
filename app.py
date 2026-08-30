from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # Expected feature order:
        # [gender, age, hypertension, heart_disease,
        #  smoking_history, bmi, HbA1c_level, blood_glucose_level]

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

        # Create feature array in the exact order used during training
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

        # Scale features using the same scaler used during training
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Get probability of class 1
        probability = model.predict_proba(
            scaled_features
        )[0][1]

        # Return result
        return jsonify({
            "diabetes_prediction": int(prediction),
            "probability": round(float(probability), 4)
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
