"""Detect trending topics using TF-IDF + K-Means clustering."""

import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.utils import PROCESSED_DIR

logger = logging.getLogger(__name__)


def detect_trends(posts: pd.DataFrame, n_clusters: int = 15) -> pd.DataFrame:
    """Cluster posts into trend topics using TF-IDF on title + selftext.

    Adds 'trend_cluster' and 'trend_label' columns to the posts DataFrame.
    """
    docs = (posts["title"].fillna("") + " " + posts["selftext"].fillna("")).tolist()
    n_clusters = min(n_clusters, len(docs))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2 if len(docs) > 10 else 1,
        max_df=0.8,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    labels = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-3:][::-1]
        top_terms = [feature_names[j] for j in top_indices]
        labels[i] = ", ".join(top_terms)

    result = posts.copy()
    result["trend_cluster"] = clusters
    result["trend_label"] = result["trend_cluster"].map(labels)

    for cluster_id, label in labels.items():
        count = (clusters == cluster_id).sum()
        logger.info("  Trend %d (%d posts): %s", cluster_id, count, label)

    return result


def assign_users_to_trends(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Map each user to the trend clusters they participated in.

    Returns a DataFrame with columns: author, trend_cluster, trend_label, first_seen_utc
    """
    post_users = posts[["author", "trend_cluster", "trend_label", "created_utc"]].copy()
    post_users = post_users.rename(columns={"created_utc": "first_seen_utc"})

    comment_users = comments.merge(
        posts[["post_id", "trend_cluster", "trend_label"]],
        on="post_id",
        how="left",
    )
    comment_users = comment_users[["author", "trend_cluster", "trend_label", "created_utc"]].copy()
    comment_users = comment_users.rename(columns={"created_utc": "first_seen_utc"})

    all_users = pd.concat([post_users, comment_users], ignore_index=True)
    all_users = all_users.dropna(subset=["trend_cluster"])
    all_users = (
        all_users.sort_values("first_seen_utc")
        .groupby(["author", "trend_cluster"], as_index=False)
        .first()
    )

    return all_users


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    posts = pd.read_csv(PROCESSED_DIR / "posts.csv")
    comments = pd.read_csv(PROCESSED_DIR / "comments.csv")

    posts = detect_trends(posts)
    user_trends = assign_users_to_trends(posts, comments)

    posts.to_csv(PROCESSED_DIR / "posts_with_trends.csv", index=False)
    user_trends.to_csv(PROCESSED_DIR / "user_trends.csv", index=False)
    logger.info("Saved %d trend assignments", len(user_trends))
