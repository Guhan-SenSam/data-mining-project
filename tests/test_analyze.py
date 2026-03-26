import pandas as pd
import networkx as nx
from src.analyze import compute_propagation_metrics


def test_compute_propagation_metrics():
    user_trends = pd.DataFrame([
        {"author": "alice", "trend_cluster": 0, "trend_label": "running", "first_seen_utc": 1000},
        {"author": "bob", "trend_cluster": 0, "trend_label": "running", "first_seen_utc": 1100},
        {"author": "carol", "trend_cluster": 0, "trend_label": "running", "first_seen_utc": 5000},
        {"author": "dave", "trend_cluster": 1, "trend_label": "study", "first_seen_utc": 2000},
    ])
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "alice",
         "title": "Running", "selftext": "", "score": 10, "num_comments": 1,
         "created_utc": 1000, "trend_cluster": 0, "trend_label": "running"},
        {"post_id": "p2", "subreddit": "Fitness", "author": "bob",
         "title": "Running too", "selftext": "", "score": 20, "num_comments": 0,
         "created_utc": 1100, "trend_cluster": 0, "trend_label": "running"},
        {"post_id": "p3", "subreddit": "GetStudying", "author": "dave",
         "title": "Study", "selftext": "", "score": 5, "num_comments": 0,
         "created_utc": 2000, "trend_cluster": 1, "trend_label": "study"},
    ])
    G = nx.DiGraph()
    G.add_edge("bob", "alice", weight=1)
    G.add_edge("carol", "bob", weight=1)

    metrics = compute_propagation_metrics(user_trends, posts, G)
    assert len(metrics) == 2

    running = [m for m in metrics if m["trend_cluster"] == 0][0]
    assert running["total_users"] == 3
    assert running["num_subreddits"] == 1
    assert running["first_appearance"] == 1000


def test_metrics_include_required_fields():
    user_trends = pd.DataFrame([
        {"author": "alice", "trend_cluster": 0, "trend_label": "test", "first_seen_utc": 1000},
    ])
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "sub1", "author": "alice",
         "title": "Test", "selftext": "", "score": 5, "num_comments": 0,
         "created_utc": 1000, "trend_cluster": 0, "trend_label": "test"},
    ])
    G = nx.DiGraph()

    metrics = compute_propagation_metrics(user_trends, posts, G)
    required = {"trend_cluster", "trend_label", "total_users", "first_appearance",
                "users_first_12h", "growth_rate_12h", "mean_degree_centrality_early",
                "mean_degree_centrality_late", "num_subreddits", "mean_post_score",
                "mean_comment_depth"}
    assert required.issubset(set(metrics[0].keys()))
