"""Train virality prediction models on trend propagation features."""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from src.utils import RESULTS_DIR

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "users_first_12h",
    "growth_rate_12h",
    "mean_degree_centrality_early",
    "num_subreddits",
    "mean_post_score",
    "mean_comment_depth",
]


def build_feature_matrix(metrics: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X and binary target y from propagation metrics.

    Target: viral (1) if total_users >= median, else not-viral (0).
    """
    df = pd.DataFrame(metrics)
    median_reach = df["total_users"].median()
    y = (df["total_users"] >= median_reach).astype(int).values
    X = df[FEATURE_COLS].values
    return X, y, FEATURE_COLS


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict:
    """Train Logistic Regression and Random Forest with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=5
        ),
    }

    min_class = min(np.bincount(y))
    n_folds = min(5, min_class) if min_class >= 2 else 2

    results = {}
    for name, model in models.items():
        X_input = X_scaled if name == "logistic_regression" else X
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        cv_results = cross_validate(
            model, X_input, y, cv=n_folds, scoring=scoring, return_estimator=True
        )

        estimator = cv_results["estimator"][-1]
        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance = np.abs(estimator.coef_[0])
        else:
            importance = np.zeros(len(feature_names))

        feature_importance = dict(zip(feature_names, [round(float(v), 4) for v in importance]))

        results[name] = {
            "mean_accuracy": round(float(cv_results["test_accuracy"].mean()), 4),
            "mean_precision": round(float(cv_results["test_precision_macro"].mean()), 4),
            "mean_recall": round(float(cv_results["test_recall_macro"].mean()), 4),
            "mean_f1": round(float(cv_results["test_f1_macro"].mean()), 4),
            "feature_importance": feature_importance,
        }
        logger.info("%s — Accuracy: %.4f, F1: %.4f", name, results[name]["mean_accuracy"], results[name]["mean_f1"])

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open(RESULTS_DIR / "propagation_metrics.json") as f:
        metrics = json.load(f)

    X, y, feature_names = build_feature_matrix(metrics)
    logger.info("Feature matrix: %d samples, %d features", X.shape[0], X.shape[1])
    logger.info("Class distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

    results = train_and_evaluate(X, y, feature_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved model results to results/metrics.json")
