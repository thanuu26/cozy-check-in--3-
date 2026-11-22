
import json
from pathlib import Path

from inference import model_fn, input_fn, predict_fn, output_fn

BASE_DIR = Path(__file__).resolve().parent

# Load the saved model
bundle = model_fn(str(BASE_DIR))

# Sample student input
sample_input = {
    "sleep": 7,
    "study": 5,
    "screen": 6,
    "meals": 3,
    "activity": 30,
    "mood": 4
}

# Simulate an API request
body = json.dumps(sample_input)

data = input_fn(body)

# Generate prediction
prediction = predict_fn(data, bundle)

# Format response
response, content_type = output_fn(prediction)

print("\n===== COZY CHECK-IN PREDICTION =====")
print("Content type:", content_type)
print(json.dumps(json.loads(response), indent=4))
