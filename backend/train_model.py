
"""
Train and compare two student wellbeing regression models.

Input:
    student_data_filled.csv

Output:
    model.joblib
    model.tar.gz
    model_results.json
"""

import argparse
import json
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score, KFold


BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "sleep",
    "study",
    "screen",
    "meals",
    "activity",
    "mood"
]

TARGET = "wellbeing"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(BASE_DIR / "student_data_filled.csv")
    )

    parser.add_argument(
        "--source",
        default="Kaggle student lifestyle dataset with generated features and target"
    )

    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.data)

    required_columns = FEATURES + [TARGET]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required_columns)

    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(float)

    print("\n===== DATASET INFORMATION =====")
    print("Dataset shape:", data.shape)
    print("Input features:", FEATURES)
    print("Target:", TARGET)

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Define two regression models
    models = {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {
        "data_source": args.source,
        "total_rows": int(len(data)),
        "features": FEATURES,
        "target": TARGET,
        "sklearn_version": sklearn.__version__,
        "models": {}
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Train and evaluate both models
    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2"
        )

        results["models"][name] = {
            "rmse": round(float(rmse), 3),
            "mae": round(float(mae), 3),
            "r2_test": round(float(r2), 3),
            "r2_cv5_mean": round(float(cv_scores.mean()), 3),
            "r2_cv5_std": round(float(cv_scores.std()), 3)
        }

        print("RMSE:", round(float(rmse), 3))
        print("MAE:", round(float(mae), 3))
        print("R2 Score:", round(float(r2), 3))
        print("5-Fold CV R2:", round(float(cv_scores.mean()), 3))

    # Select the model with the lowest test RMSE
    best_name = min(
        results["models"],
        key=lambda name: results["models"][name]["rmse"]
    )

    print("\n===== MODEL COMPARISON =====")
    print("Best model:", best_name)

    # Retrain selected model using the complete dataset
    best_model = models[best_name]
    best_model.fit(X, y)

    # Calculate model-agnostic feature importance
    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )

    importance = {
        feature: round(float(value), 4)
        for feature, value in zip(
            FEATURES,
            importance_result.importances_mean
        )
    }

    # Save trained model and metadata
    output_dir = BASE_DIR

    model_path = output_dir / "model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "target": TARGET,
            "model_name": best_name,
            "importance": importance
        },
        model_path
    )

    # Package model for deployment
    archive_path = output_dir / "model.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(
            model_path,
            arcname="model.joblib"
        )

    results["best"] = best_name
    results["feature_importance"] = importance

    results_path = output_dir / "model_results.json"

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\n===== FEATURE IMPORTANCE =====")

    for feature, value in importance.items():
        print(feature, ":", value)

    print("\n===== FILES GENERATED =====")
    print("Model:", model_path)
    print("AWS package:", archive_path)
    print("Results:", results_path)

    print("\nTRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
