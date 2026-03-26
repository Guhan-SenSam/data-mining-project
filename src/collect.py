"""Collect posts and comments from Reddit using the .json API."""

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils import fetch_json, RAW_DIR, PROCESSED_DIR, DATA_DIR

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"


def parse_post(data: dict, subreddit: str) -> dict:
    """Extract relevant fields from a Reddit post."""
    return {
        "post_id": data["id"],
        "subreddit": subreddit,
        "author": data.get("author", "[deleted]"),
        "title": data.get("title", ""),
        "selftext": data.get("selftext", ""),
        "score": data.get("score", 0),
        "num_comments": data.get("num_comments", 0),
        "created_utc": data.get("created_utc", 0),
    }


def parse_comment(data: dict, post_id: str) -> dict:
    """Extract relevant fields from a Reddit comment."""
    return {
        "comment_id": data["id"],
        "post_id": post_id,
        "parent_id": data.get("parent_id", ""),
        "author": data.get("author", "[deleted]"),
        "body": data.get("body", ""),
        "score": data.get("score", 0),
        "created_utc": data.get("created_utc", 0),
    }


def flatten_comment_tree(children: list, post_id: str) -> list[dict]:
    """Recursively flatten a Reddit comment tree into a list of comments."""
    comments = []
    for child in children:
        if child["kind"] != "t1":
            continue
        data = child["data"]
        comments.append(parse_comment(data, post_id))
        replies = data.get("replies")
        if replies and isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            comments.extend(flatten_comment_tree(reply_children, post_id))
    return comments


def fetch_posts(subreddit: str, sort: str = "new", limit: int = 25, max_pages: int = 1) -> list[dict]:
    """Fetch posts from a subreddit, paginating through results."""
    posts = []
    after = None
    for _ in range(max_pages):
        url = f"{REDDIT_BASE}/r/{subreddit}/{sort}.json?limit={limit}"
        if after:
            url += f"&after={after}"
        try:
            data = fetch_json(url)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            break
        children = data.get("data", {}).get("children", [])
        for child in children:
            posts.append(parse_post(child["data"], subreddit))
        after = data.get("data", {}).get("after")
        if not after:
            break
    return posts


def fetch_comments(subreddit: str, post_id: str) -> list[dict]:
    """Fetch all comments for a post."""
    url = f"{REDDIT_BASE}/r/{subreddit}/comments/{post_id}.json"
    try:
        data = fetch_json(url)
    except Exception as e:
        logger.warning("Failed to fetch comments for %s: %s", post_id, e)
        return []
    if not isinstance(data, list) or len(data) < 2:
        return []
    comment_children = data[1].get("data", {}).get("children", [])
    return flatten_comment_tree(comment_children, post_id)


def collect_subreddit(subreddit: str) -> tuple[list[dict], list[dict]]:
    """Collect all posts and comments from a subreddit."""
    logger.info("Collecting r/%s ...", subreddit)
    all_posts = []
    all_comments = []

    for sort in ("hot",):
        posts = fetch_posts(subreddit, sort=sort)
        all_posts.extend(posts)

    # Deduplicate posts by post_id
    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["post_id"] not in seen:
            seen.add(p["post_id"])
            unique_posts.append(p)
    all_posts = unique_posts

    # Fetch comments for each post
    for post in all_posts:
        comments = fetch_comments(subreddit, post["post_id"])
        all_comments.extend(comments)

    logger.info("  r/%s: %d posts, %d comments", subreddit, len(all_posts), len(all_comments))
    return all_posts, all_comments


def save_raw(subreddit: str, posts: list[dict], comments: list[dict]) -> None:
    """Save raw data as JSON for reproducibility."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{subreddit}.json"
    raw_path.write_text(json.dumps({"posts": posts, "comments": comments}, indent=2))


def run_collection() -> None:
    """Run full collection pipeline across all discovered subreddits."""
    subreddits_path = DATA_DIR / "subreddits.json"
    if not subreddits_path.exists():
        raise FileNotFoundError("Run discover first: uv run python -m src.discover")

    subreddits = json.loads(subreddits_path.read_text())
    all_posts = []
    all_comments = []

    for sub in subreddits:
        posts, comments = collect_subreddit(sub)
        save_raw(sub, posts, comments)
        all_posts.extend(posts)
        all_comments.extend(comments)

    # Save processed CSVs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_posts).to_csv(PROCESSED_DIR / "posts.csv", index=False)
    pd.DataFrame(all_comments).to_csv(PROCESSED_DIR / "comments.csv", index=False)
    logger.info("Saved %d posts and %d comments", len(all_posts), len(all_comments))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_collection()
