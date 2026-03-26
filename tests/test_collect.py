import json
from unittest.mock import patch, MagicMock
from src.collect import (
    fetch_posts,
    fetch_comments,
    parse_post,
    parse_comment,
    flatten_comment_tree,
)


SAMPLE_POST = {
    "kind": "t3",
    "data": {
        "id": "abc123",
        "subreddit": "Fitness",
        "author": "testuser",
        "title": "Started running today",
        "selftext": "Just did my first 5k",
        "score": 42,
        "num_comments": 5,
        "created_utc": 1711324800.0,
    },
}

SAMPLE_COMMENT = {
    "kind": "t1",
    "data": {
        "id": "com456",
        "link_id": "t3_abc123",
        "parent_id": "t3_abc123",
        "author": "replyer",
        "body": "Great job!",
        "score": 10,
        "created_utc": 1711328400.0,
    },
}


def test_parse_post():
    result = parse_post(SAMPLE_POST["data"], "Fitness")
    assert result["post_id"] == "abc123"
    assert result["subreddit"] == "Fitness"
    assert result["author"] == "testuser"
    assert result["title"] == "Started running today"


def test_parse_comment():
    result = parse_comment(SAMPLE_COMMENT["data"], "abc123")
    assert result["comment_id"] == "com456"
    assert result["post_id"] == "abc123"
    assert result["author"] == "replyer"


def test_flatten_comment_tree_skips_more():
    tree = [
        {"kind": "t1", "data": {"id": "c1", "link_id": "t3_p1", "parent_id": "t3_p1",
         "author": "u1", "body": "hi", "score": 1, "created_utc": 1.0, "replies": ""}},
        {"kind": "more", "data": {"children": ["c2"]}},
    ]
    result = flatten_comment_tree(tree, "p1")
    assert len(result) == 1
    assert result[0]["comment_id"] == "c1"


def test_fetch_posts_paginates():
    page1 = {"data": {"children": [SAMPLE_POST], "after": "t3_next"}}
    page2 = {"data": {"children": [SAMPLE_POST], "after": None}}

    with patch("src.collect.fetch_json", side_effect=[page1, page2]):
        posts = fetch_posts("Fitness", sort="new", limit=2, max_pages=2)

    assert len(posts) == 2
