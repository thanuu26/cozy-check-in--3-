
"""
Cozy Check-In inference script.

Supports:
1. Local model prediction
2. AWS SageMaker SKLearnModel inference
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    "sleep",
    "study",
    "screen",
    "meals",
    "activity",
    "mood"
]

# Reference values used for habit feedback.
IDEAL = {
    "sleep": 8,
    "study": 4,
    "screen": 3,
    "meals": 3,
    "activity": 45,
    "mood": 5
}

SPAN = {
    "sleep": 4,
    "study": 6,
    "screen": 7,
    "meals": 3,
    "activity": 45,
    "mood": 4
}

SIDE = {
    "sleep": "both",
    "study": "above",
    "screen": "above",
    "meals": "both",
    "activity": "below",
    "mood": "below"
}

NAMES = {
    "sleep": "Sleep",
    "study": "Study load",
    "screen": "Screen time",
    "meals": "Meals",
    "activity": "Physical activity",
    "mood": "Mood"
}

TIPS = {
    "sleep": "Aim for 7 to 9 hours of sleep.",
    "study": "Take short breaks during long study sessions.",
    "screen": "Try reducing unnecessary screen time.",
    "meals": "Try to maintain regular meal timings.",
    "activity": "Include a short walk or physical activity.",
    "mood": "Consider talking to someone you trust."
}


# Load the trained model
def model_fn(model_dir):

    model_path = os.path.join(
        model_dir,
        "model.joblib"
    )

    bundle = joblib.load(model_path)

    return bundle


# Read incoming JSON request
def input_fn(body, content_type="application/json"):

    if content_type != "application/json":
        raise ValueError("Only application/json is supported.")

    if isinstance(body, bytes):
        body = body.decode("utf-8")

    data = json.loads(body)

    return data


# Calculate deviation from a reference habit value
def calculate_gap(feature, value):

    difference = value - IDEAL[feature]

    if SIDE[feature] == "both":
        return abs(difference)

    elif SIDE[feature] == "above":
        return max(difference, 0)

    else:
        return max(-difference, 0)


# Generate prediction and habit recommendations
def predict_fn(data, bundle):

    missing = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    values = {
        feature: float(data[feature])
        for feature in FEATURES
    }

    # Validate user input ranges
    limits = {
        "sleep": (0, 12),
        "study": (0, 12),
        "screen": (0, 24),
        "meals": (0, 5),
        "activity": (0, 180),
        "mood": (1, 5)
    }

    for feature, value in values.items():

        minimum, maximum = limits[feature]

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between "
                f"{minimum} and {maximum}."
            )

    # Prepare model input in training feature order
    X = pd.DataFrame(
        [[values[f] for f in FEATURES]],
        columns=FEATURES
    )

    # Predict wellbeing score
    prediction = bundle["model"].predict(X)[0]

    score = float(
        np.clip(prediction, 0, 100)
    )

    # Calculate habit deviations
    importance = bundle["importance"]

    gaps = {}

    for feature in FEATURES:

        weight = max(
            float(importance.get(feature, 0)),
            0
        )

        deviation = min(
            calculate_gap(feature, values[feature])
            / SPAN[feature],
            1
        )

        gaps[feature] = weight * deviation

    # Rank habits by estimated recommendation priority
    ranked = sorted(
        gaps.items(),
        key=lambda item: item[1],
        reverse=True
    )

    factors = []

    for feature, priority in ranked[:3]:

        if priority <= 0:
            continue

        factors.append({
            "name": NAMES[feature],
            "priority": round(float(priority), 3),
            "tip": TIPS[feature]
        })

    return {
        "score": round(score, 1),
        "factors": factors,
        "message": "This is a general wellbeing estimate, not a medical assessment."
    }


# Return JSON response
def output_fn(prediction, accept="application/json"):

    return (
        json.dumps(prediction),
        "application/json"
    )
