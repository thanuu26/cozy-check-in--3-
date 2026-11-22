
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from inference import model_fn, input_fn, predict_fn, output_fn

app = Flask(__name__)
CORS(app)

# Load the trained model once when the server starts
BASE_DIR = Path(__file__).resolve().parent
model_bundle = model_fn(str(BASE_DIR))


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Cozy Check-In ML API is ready!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = input_fn(
            request.get_data(as_text=True),
            request.content_type or "application/json"
        )

        prediction = predict_fn(data, model_bundle)
        response, content_type = output_fn(prediction)

        return response, 200, {"Content-Type": content_type}

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server terminal."}), 500


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )