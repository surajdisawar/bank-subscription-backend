
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "bank_subscription_model.pkl"
)

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Bank Subscription Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        columns = [
            "age", "job", "marital", "education", "default",
            "balance", "housing", "loan", "contact", "day",
            "month", "duration", "campaign", "pdays",
            "previous", "poutcome"
        ]

        df = pd.DataFrame(
            [[data.get(col) for col in columns]],
            columns=columns
        )

        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": str(prediction),
            "message": "Prediction generated successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
