import pandas as pd
from src.trends import detect_trends, assign_users_to_trends


def make_posts():
    return pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "alice",
         "title": "Started running every morning", "selftext": "Running is great for cardio",
         "score": 10, "num_comments": 2, "created_utc": 1000},
        {"post_id": "p2", "subreddit": "Fitness", "author": "bob",
         "title": "My running routine", "selftext": "I run 5k every day",
         "score": 20, "num_comments": 3, "created_utc": 1100},
        {"post_id": "p3", "subreddit": "GetStudying", "author": "carol",
         "title": "Study schedule for exams", "selftext": "Studying 4 hours daily",
         "score": 15, "num_comments": 1, "created_utc": 1200},
        {"post_id": "p4", "subreddit": "GetStudying", "author": "dave",
         "title": "Best study techniques", "selftext": "Flashcards and spaced repetition",
         "score": 8, "num_comments": 0, "created_utc": 1300},
    ])


def test_detect_trends_returns_labeled_posts():
    posts = make_posts()
    result = detect_trends(posts, n_clusters=2)
    assert "trend_cluster" in result.columns
    assert "trend_label" in result.columns
    assert len(result) == 4
    assert result["trend_cluster"].nunique() == 2


def test_assign_users_to_trends():
    posts = make_posts()
    posts["trend_cluster"] = [0, 0, 1, 1]
    posts["trend_label"] = ["running", "running", "study", "study"]

    comments = pd.DataFrame([
        {"comment_id": "c1", "post_id": "p1", "parent_id": "t3_p1",
         "author": "eve", "body": "Nice!", "score": 1, "created_utc": 1050},
        {"comment_id": "c2", "post_id": "p3", "parent_id": "t3_p3",
         "author": "eve", "body": "Same!", "score": 1, "created_utc": 1250},
    ])

    user_trends = assign_users_to_trends(posts, comments)
    eve_trends = user_trends[user_trends["author"] == "eve"]
    assert len(eve_trends["trend_cluster"].unique()) == 2
