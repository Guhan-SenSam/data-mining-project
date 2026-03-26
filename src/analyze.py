"""Compute propagation metrics per trend cluster."""

import json
import logging

import networkx as nx
import pandas as pd

from src.utils import PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

TWELVE_HOURS = 12 * 3600


def _mean_degree_centrality(G: nx.DiGraph, users: list[str]) -> float:
    """Compute mean degree centrality for a subset of users in the graph."""
    if not users or G.number_of_nodes() == 0:
        return 0.0
    centrality = nx.degree_centrality(G)
    values = [centrality.get(u, 0.0) for u in users]
    return sum(values) / len(values) if values else 0.0


def _mean_comment_depth(posts: pd.DataFrame, comments: pd.DataFrame) -> float:
    """Compute average reply chain depth for comments in the given posts."""
    if comments is None or comments.empty:
        return 0.0
    reply_count = comments["parent_id"].str.startswith("t1_").sum()
    return reply_count / len(comments) if len(comments) > 0 else 0.0


def compute_propagation_metrics(
    user_trends: pd.DataFrame,
    posts: pd.DataFrame,
    G: nx.DiGraph,
) -> list[dict]:
    """Compute propagation metrics for each trend cluster."""
    comments_path = PROCESSED_DIR / "comments.csv"
    comments = pd.read_csv(comments_path) if comments_path.exists() else pd.DataFrame()

    metrics = []
    for cluster_id, group in user_trends.groupby("trend_cluster"):
        label = group["trend_label"].iloc[0]
        sorted_users = group.sort_values("first_seen_utc")
        total_users = len(sorted_users)

        first_appearance = sorted_users["first_seen_utc"].min()
        cutoff_12h = first_appearance + TWELVE_HOURS

        early_users = sorted_users[sorted_users["first_seen_utc"] <= cutoff_12h]["author"].tolist()
        late_users = sorted_users[sorted_users["first_seen_utc"] > cutoff_12h]["author"].tolist()
        users_first_12h = len(early_users)

        growth_rate_12h = users_first_12h / 12.0 if users_first_12h > 0 else 0.0

        mean_cent_early = _mean_degree_centrality(G, early_users)
        mean_cent_late = _mean_degree_centrality(G, late_users)

        cluster_posts = posts[posts["trend_cluster"] == cluster_id]
        num_subreddits = cluster_posts["subreddit"].nunique()
        mean_post_score = cluster_posts["score"].mean() if len(cluster_posts) > 0 else 0.0

        cluster_post_ids = set(cluster_posts["post_id"])
        cluster_comments = comments[comments["post_id"].isin(cluster_post_ids)] if not comments.empty else pd.DataFrame()
        mean_depth = _mean_comment_depth(cluster_posts, cluster_comments)

        metrics.append({
            "trend_cluster": int(cluster_id),
            "trend_label": label,
            "total_users": total_users,
            "first_appearance": float(first_appearance),
            "users_first_12h": users_first_12h,
            "growth_rate_12h": round(growth_rate_12h, 4),
            "mean_degree_centrality_early": round(mean_cent_early, 6),
            "mean_degree_centrality_late": round(mean_cent_late, 6),
            "num_subreddits": num_subreddits,
            "mean_post_score": round(mean_post_score, 2),
            "mean_comment_depth": round(mean_depth, 4),
        })

    logger.info("Computed metrics for %d trend clusters", len(metrics))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    posts = pd.read_csv(PROCESSED_DIR / "posts_with_trends.csv")
    user_trends = pd.read_csv(PROCESSED_DIR / "user_trends.csv")
    G = nx.read_gexf(str(PROCESSED_DIR / "interaction_graph.gexf"))

    metrics = compute_propagation_metrics(user_trends, posts, G)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "propagation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved propagation metrics to results/propagation_metrics.json")
