import pandas as pd
from src.predict import build_feature_matrix, train_and_evaluate


def make_metrics():
    """Create sample propagation metrics for testing."""
    return [
        {"trend_cluster": i, "trend_label": f"trend_{i}", "total_users": 10 + i * 5,
         "first_appearance": 1000.0, "users_first_12h": 3 + i,
         "growth_rate_12h": 0.25 + i * 0.1, "mean_degree_centrality_early": 0.01 * i,
         "mean_degree_centrality_late": 0.005 * i, "num_subreddits": 1 + i % 3,
         "mean_post_score": 10.0 + i, "mean_comment_depth": 0.3 + i * 0.05}
        for i in range(20)
    ]


def test_build_feature_matrix():
    metrics = make_metrics()
    X, y, feature_names = build_feature_matrix(metrics)
    assert X.shape[0] == 20
    assert X.shape[1] == 6
    assert len(y) == 20
    assert set(y).issubset({0, 1})
    assert len(feature_names) == 6


def test_train_and_evaluate():
    metrics = make_metrics()
    X, y, feature_names = build_feature_matrix(metrics)
    results = train_and_evaluate(X, y, feature_names)
    assert "logistic_regression" in results
    assert "random_forest" in results
    for model_name, model_results in results.items():
        assert "mean_accuracy" in model_results
        assert "mean_f1" in model_results
        assert "feature_importance" in model_results
