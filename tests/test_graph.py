import pandas as pd
import networkx as nx
from src.graph import build_interaction_graph, EXCLUDED_AUTHORS


def test_build_graph_creates_edges():
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "alice",
         "title": "Running", "selftext": "", "score": 10, "num_comments": 1, "created_utc": 1000},
    ])
    comments = pd.DataFrame([
        {"comment_id": "c1", "post_id": "p1", "parent_id": "t3_p1",
         "author": "bob", "body": "Nice!", "score": 5, "created_utc": 1100},
    ])
    G = build_interaction_graph(posts, comments)
    assert G.has_node("alice")
    assert G.has_node("bob")
    assert G.has_edge("bob", "alice")


def test_build_graph_excludes_deleted():
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "[deleted]",
         "title": "Test", "selftext": "", "score": 1, "num_comments": 1, "created_utc": 1000},
    ])
    comments = pd.DataFrame([
        {"comment_id": "c1", "post_id": "p1", "parent_id": "t3_p1",
         "author": "bob", "body": "Hi", "score": 1, "created_utc": 1100},
    ])
    G = build_interaction_graph(posts, comments)
    assert "[deleted]" not in G.nodes


def test_build_graph_excludes_automoderator():
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "alice",
         "title": "Test", "selftext": "", "score": 1, "num_comments": 1, "created_utc": 1000},
    ])
    comments = pd.DataFrame([
        {"comment_id": "c1", "post_id": "p1", "parent_id": "t3_p1",
         "author": "AutoModerator", "body": "Removed", "score": 1, "created_utc": 1100},
    ])
    G = build_interaction_graph(posts, comments)
    assert "AutoModerator" not in G.nodes


def test_build_graph_handles_reply_chains():
    posts = pd.DataFrame([
        {"post_id": "p1", "subreddit": "Fitness", "author": "alice",
         "title": "Test", "selftext": "", "score": 1, "num_comments": 2, "created_utc": 1000},
    ])
    comments = pd.DataFrame([
        {"comment_id": "c1", "post_id": "p1", "parent_id": "t3_p1",
         "author": "bob", "body": "Hi", "score": 1, "created_utc": 1100},
        {"comment_id": "c2", "post_id": "p1", "parent_id": "t1_c1",
         "author": "carol", "body": "Hey", "score": 1, "created_utc": 1200},
    ])
    G = build_interaction_graph(posts, comments)
    assert G.has_edge("carol", "bob")
    assert G.has_edge("bob", "alice")
