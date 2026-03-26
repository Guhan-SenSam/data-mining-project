# Social Contagion of Trends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that discovers trending subreddits, collects Reddit data, constructs a user interaction graph, detects trends, and predicts trend virality using ML.

**Architecture:** 4-phase pipeline (discover → collect → graph+trends → analyze+predict). Each phase is a standalone module runnable via `uv run python -m src.<module>`. Data flows through `data/` directory as JSON and CSV files.

**Tech Stack:** Python, UV, requests, NetworkX, scikit-learn, pandas, matplotlib, seaborn, google-generativeai

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | UV project config, dependencies |
| `src/__init__.py` | Package marker |
| `src/utils.py` | Rate-limited HTTP client, config constants, path helpers |
| `src/discover.py` | Gemini API call to get subreddit list |
| `src/collect.py` | Reddit `.json` scraping, raw JSON storage, CSV parsing |
| `src/graph.py` | NetworkX directed graph from comments |
| `src/trends.py` | TF-IDF + K-Means trend clustering |
| `src/analyze.py` | Propagation metrics per trend cluster |
| `src/predict.py` | Virality prediction (LogReg + RF), evaluation |
| `notebooks/analysis.ipynb` | Visualizations and analysis narrative |
| `tests/test_utils.py` | Tests for utils |
| `tests/test_discover.py` | Tests for discover |
| `tests/test_collect.py` | Tests for collect |
| `tests/test_graph.py` | Tests for graph |
| `tests/test_trends.py` | Tests for trends |
| `tests/test_analyze.py` | Tests for analyze |
| `tests/test_predict.py` | Tests for predict |

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize UV project**

```bash
cd /mnt/Dev/Projects/data-mining-project
uv init --name social-contagion-trends --python 3.12
```

- [ ] **Step 2: Add dependencies**

```bash
uv add requests networkx scikit-learn pandas matplotlib seaborn google-generativeai
uv add --dev pytest
```

- [ ] **Step 3: Create package structure**

Create `src/__init__.py`:
```python
"""Social Contagion of Trends — Reddit data mining pipeline."""
```

Create `tests/__init__.py`:
```python
```

```bash
mkdir -p data/raw data/processed results/figures notebooks
```

- [ ] **Step 4: Verify setup**

```bash
uv run python -c "import requests, networkx, sklearn, pandas, matplotlib, seaborn; print('All imports OK')"
uv run pytest --co
```

Expected: "All imports OK" and pytest collects 0 tests.

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml uv.lock src/ tests/ CLAUDE.md docs/
git commit -m "chore: scaffold project with UV, add dependencies"
```

---

### Task 2: Utility Module — Rate-Limited HTTP Client & Config

**Files:**
- Create: `src/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing test for rate-limited fetcher**

Create `tests/test_utils.py`:
```python
import time
from unittest.mock import patch, MagicMock
from src.utils import fetch_json, DATA_DIR, RAW_DIR, PROCESSED_DIR


def test_fetch_json_returns_parsed_json():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"kind": "Listing", "data": {}}
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.requests.get", return_value=mock_response) as mock_get:
        result = fetch_json("https://example.com/test.json")

    assert result == {"kind": "Listing", "data": {}}
    mock_get.assert_called_once()
    call_headers = mock_get.call_args[1]["headers"]
    assert "User-Agent" in call_headers


def test_fetch_json_respects_rate_limit():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.requests.get", return_value=mock_response):
        start = time.time()
        fetch_json("https://example.com/a.json")
        fetch_json("https://example.com/b.json")
        elapsed = time.time() - start

    assert elapsed >= 1.0, "Should wait at least 1s between requests"


def test_data_dirs_exist():
    assert DATA_DIR.name == "data"
    assert RAW_DIR.name == "raw"
    assert PROCESSED_DIR.name == "processed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils'`

- [ ] **Step 3: Implement utils.py**

Create `src/utils.py`:
```python
"""Shared utilities: rate-limited HTTP client, path constants."""

import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

USER_AGENT = "SocialContagionTrends/1.0 (academic research project)"
_last_request_time = 0.0
RATE_LIMIT_SECONDS = 1.5


def fetch_json(url: str) -> dict:
    """Fetch a URL and return parsed JSON, respecting rate limits."""
    global _last_request_time
    now = time.time()
    wait = RATE_LIMIT_SECONDS - (now - _last_request_time)
    if wait > 0:
        time.sleep(wait)

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request_time = time.time()
    response.raise_for_status()
    return response.json()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add rate-limited HTTP client and path constants"
```

---

### Task 3: Subreddit Discovery via Gemini API

**Files:**
- Create: `src/discover.py`
- Create: `tests/test_discover.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_discover.py`:
```python
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.discover import discover_subreddits, SUBREDDITS_PATH


def test_discover_subreddits_returns_list():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        "Fitness", "GetStudying", "productivity", "loseit",
        "Meditation", "getdisciplined"
    ])
    mock_model.generate_content.return_value = mock_response

    with patch("src.discover.genai") as mock_genai:
        mock_genai.GenerativeModel.return_value = mock_model
        result = discover_subreddits()

    assert isinstance(result, list)
    assert len(result) >= 6
    assert "Fitness" in result


def test_discover_uses_cache(tmp_path):
    cache_file = tmp_path / "subreddits.json"
    cached = ["Fitness", "running", "loseit"]
    cache_file.write_text(json.dumps(cached))

    with patch("src.discover.SUBREDDITS_PATH", cache_file):
        result = discover_subreddits()

    assert result == cached
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_discover.py -v
```

Expected: FAIL — cannot import `discover_subreddits`.

- [ ] **Step 3: Implement discover.py**

Create `src/discover.py`:
```python
"""Discover trending habit-related subreddits using Gemini API."""

import json
import os

import google.generativeai as genai

from src.utils import DATA_DIR

SUBREDDITS_PATH = DATA_DIR / "subreddits.json"

PROMPT = """List 25-30 active Reddit subreddit names (without the r/ prefix) where people
discuss habits, self-improvement, and lifestyle changes. Include categories like:
- Fitness and exercise (e.g., Fitness, running, bodyweightfitness)
- Study and learning (e.g., GetStudying, studytips)
- Productivity (e.g., productivity, getdisciplined)
- Diet and nutrition (e.g., loseit, EatCheapAndHealthy)
- Mental health and mindfulness (e.g., Meditation, selfimprovement)
- Sleep (e.g., sleep, SleepApnea)
- Creative habits (e.g., writing, drawing)

Return ONLY a JSON array of subreddit names, no other text. Example: ["Fitness", "running"]"""


def discover_subreddits() -> list[str]:
    """Return list of subreddit names, using cache if available."""
    if SUBREDDITS_PATH.exists():
        return json.loads(SUBREDDITS_PATH.read_text())

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(PROMPT)

    subreddits = json.loads(response.text)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUBREDDITS_PATH.write_text(json.dumps(subreddits, indent=2))
    return subreddits


if __name__ == "__main__":
    subs = discover_subreddits()
    print(f"Discovered {len(subs)} subreddits: {subs}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_discover.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/discover.py tests/test_discover.py
git commit -m "feat: add Gemini-powered subreddit discovery with caching"
```

---

### Task 4: Reddit Data Collection

**Files:**
- Create: `src/collect.py`
- Create: `tests/test_collect.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_collect.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_collect.py -v
```

Expected: FAIL — cannot import from `src.collect`.

- [ ] **Step 3: Implement collect.py**

Create `src/collect.py`:
```python
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


def fetch_posts(subreddit: str, sort: str = "new", limit: int = 100, max_pages: int = 4) -> list[dict]:
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

    for sort in ("hot", "new"):
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_collect.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/collect.py tests/test_collect.py
git commit -m "feat: add Reddit data collection with pagination and comment tree flattening"
```

---

### Task 5: Graph Construction

**Files:**
- Create: `src/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph.py`:
```python
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
    # carol replied to bob's comment
    assert G.has_edge("carol", "bob")
    # bob replied to alice's post
    assert G.has_edge("bob", "alice")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: FAIL — cannot import from `src.graph`.

- [ ] **Step 3: Implement graph.py**

Create `src/graph.py`:
```python
"""Build user interaction graph from Reddit posts and comments."""

import logging

import networkx as nx
import pandas as pd

from src.utils import PROCESSED_DIR

logger = logging.getLogger(__name__)

EXCLUDED_AUTHORS = {"[deleted]", "AutoModerator"}


def build_interaction_graph(posts: pd.DataFrame, comments: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph where edges represent comment interactions.

    Edges go from commenter -> post author (for top-level comments)
    and from commenter -> parent comment author (for replies).
    """
    G = nx.DiGraph()

    # Build lookup: post_id -> author
    post_authors = dict(zip(posts["post_id"], posts["author"]))

    # Build lookup: comment_id -> author
    comment_authors = dict(zip(comments["comment_id"], comments["author"]))

    for _, comment in comments.iterrows():
        commenter = comment["author"]
        if commenter in EXCLUDED_AUTHORS:
            continue

        parent_id = comment["parent_id"]
        post_id = comment["post_id"]
        timestamp = comment["created_utc"]
        subreddit = post_authors.get(post_id, "")

        # Determine who the commenter is replying to
        if parent_id.startswith("t3_"):
            # Top-level comment -> replying to post author
            target_author = post_authors.get(parent_id[3:])
        elif parent_id.startswith("t1_"):
            # Reply to another comment
            target_author = comment_authors.get(parent_id[3:])
        else:
            continue

        if not target_author or target_author in EXCLUDED_AUTHORS:
            continue
        if commenter == target_author:
            continue  # Skip self-replies

        if G.has_edge(commenter, target_author):
            G[commenter][target_author]["weight"] += 1
        else:
            G.add_edge(commenter, target_author, weight=1,
                       subreddit=subreddit, timestamp=timestamp, post_id=post_id)

    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    posts = pd.read_csv(PROCESSED_DIR / "posts.csv")
    comments = pd.read_csv(PROCESSED_DIR / "comments.csv")
    G = build_interaction_graph(posts, comments)
    nx.write_gexf(G, str(PROCESSED_DIR / "interaction_graph.gexf"))
    logger.info("Saved graph to %s", PROCESSED_DIR / "interaction_graph.gexf")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat: add user interaction graph construction from comments"
```

---

### Task 6: Trend Detection via TF-IDF + K-Means

**Files:**
- Create: `src/trends.py`
- Create: `tests/test_trends.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trends.py`:
```python
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
    # eve should appear in both trend clusters
    eve_trends = user_trends[user_trends["author"] == "eve"]
    assert len(eve_trends["trend_cluster"].unique()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_trends.py -v
```

Expected: FAIL — cannot import from `src.trends`.

- [ ] **Step 3: Implement trends.py**

Create `src/trends.py`:
```python
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
    # Combine title and selftext into a single document per post
    docs = (posts["title"].fillna("") + " " + posts["selftext"].fillna("")).tolist()

    # Adjust n_clusters if we have fewer documents
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

    # Generate labels from top TF-IDF terms per cluster
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

    A user 'adopts' a trend if they authored a post or commented on a post
    in that trend cluster.

    Returns a DataFrame with columns: author, trend_cluster, trend_label, first_seen_utc
    """
    # Post authors
    post_users = posts[["author", "trend_cluster", "trend_label", "created_utc"]].copy()
    post_users = post_users.rename(columns={"created_utc": "first_seen_utc"})

    # Comment authors: join comments with posts to get trend info
    comment_users = comments.merge(
        posts[["post_id", "trend_cluster", "trend_label"]],
        on="post_id",
        how="left",
    )
    comment_users = comment_users[["author", "trend_cluster", "trend_label", "created_utc"]].copy()
    comment_users = comment_users.rename(columns={"created_utc": "first_seen_utc"})

    # Combine and deduplicate: keep earliest appearance per (author, trend)
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_trends.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/trends.py tests/test_trends.py
git commit -m "feat: add TF-IDF + K-Means trend detection and user-trend assignment"
```

---

### Task 7: Propagation Analysis

**Files:**
- Create: `src/analyze.py`
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyze.py`:
```python
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
    assert len(metrics) == 2  # Two trend clusters

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_analyze.py -v
```

Expected: FAIL — cannot import from `src.analyze`.

- [ ] **Step 3: Implement analyze.py**

Create `src/analyze.py`:
```python
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
    # Depth = number of parent hops. t3_ parents are depth 0, t1_ parents are depth 1+
    # Approximate: count fraction of comments that are replies to other comments
    reply_count = comments["parent_id"].str.startswith("t1_").sum()
    return reply_count / len(comments) if len(comments) > 0 else 0.0


def compute_propagation_metrics(
    user_trends: pd.DataFrame,
    posts: pd.DataFrame,
    G: nx.DiGraph,
) -> list[dict]:
    """Compute propagation metrics for each trend cluster."""
    # Load comments for depth calculation
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

        # Growth rate: users per hour in first 12h
        growth_rate_12h = users_first_12h / 12.0 if users_first_12h > 0 else 0.0

        # Centrality metrics
        mean_cent_early = _mean_degree_centrality(G, early_users)
        mean_cent_late = _mean_degree_centrality(G, late_users)

        # Cross-subreddit spread
        cluster_posts = posts[posts["trend_cluster"] == cluster_id]
        num_subreddits = cluster_posts["subreddit"].nunique()
        mean_post_score = cluster_posts["score"].mean() if len(cluster_posts) > 0 else 0.0

        # Comment depth for this cluster's posts
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_analyze.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analyze.py tests/test_analyze.py
git commit -m "feat: add propagation metrics computation per trend cluster"
```

---

### Task 8: Virality Prediction Model

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_predict.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_predict.py -v
```

Expected: FAIL — cannot import from `src.predict`.

- [ ] **Step 3: Implement predict.py**

Create `src/predict.py`:
```python
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

    # Use at most 5 folds, but no more than the smallest class count
    min_class = min(np.bincount(y))
    n_folds = min(5, min_class) if min_class >= 2 else 2

    results = {}
    for name, model in models.items():
        X_input = X_scaled if name == "logistic_regression" else X
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        cv_results = cross_validate(
            model, X_input, y, cv=n_folds, scoring=scoring, return_estimator=True
        )

        # Feature importance from last fold's estimator
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_predict.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add virality prediction with LogReg and Random Forest"
```

---

### Task 9: Analysis Notebook

**Files:**
- Create: `notebooks/analysis.ipynb`

- [ ] **Step 1: Create the analysis notebook**

Create `notebooks/analysis.ipynb` with these cells:

**Cell 1 (Markdown):**
```markdown
# Social Contagion of Trends — Analysis

This notebook presents the full analysis pipeline: data overview, interaction graph, trend detection, propagation metrics, and virality prediction results.
```

**Cell 2 (Code):**
```python
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_DIR = Path("../data/processed")
RESULTS_DIR = Path("../results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

**Cell 3 (Markdown):**
```markdown
## 1. Data Overview
```

**Cell 4 (Code):**
```python
posts = pd.read_csv(DATA_DIR / "posts.csv")
comments = pd.read_csv(DATA_DIR / "comments.csv")
print(f"Posts: {len(posts)}, Comments: {len(comments)}")
print(f"Subreddits: {posts['subreddit'].nunique()}")
print(f"Unique authors (posts): {posts['author'].nunique()}")
print(f"Unique authors (comments): {comments['author'].nunique()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
posts["subreddit"].value_counts().head(15).plot.barh(ax=axes[0], title="Posts per Subreddit (Top 15)")
comments.groupby(posts.set_index("post_id").loc[comments["post_id"]]["subreddit"].values).size().sort_values().tail(15).plot.barh(ax=axes[1], title="Comments per Subreddit (Top 15)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "data_overview.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 5 (Markdown):**
```markdown
## 2. Interaction Graph
```

**Cell 6 (Code):**
```python
G = nx.read_gexf(str(DATA_DIR / "interaction_graph.gexf"))
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.6f}")

degree_seq = sorted([d for n, d in G.degree()], reverse=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(degree_seq, bins=50, edgecolor="black")
axes[0].set_title("Degree Distribution")
axes[0].set_xlabel("Degree")
axes[0].set_ylabel("Count")
axes[0].set_yscale("log")

centrality = nx.degree_centrality(G)
top_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
names, vals = zip(*top_users)
axes[1].barh(range(len(names)), vals)
axes[1].set_yticks(range(len(names)))
axes[1].set_yticklabels(names, fontsize=8)
axes[1].set_title("Top 20 Users by Degree Centrality")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "graph_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 7 (Markdown):**
```markdown
## 3. Trend Detection
```

**Cell 8 (Code):**
```python
posts_trends = pd.read_csv(DATA_DIR / "posts_with_trends.csv")
user_trends = pd.read_csv(DATA_DIR / "user_trends.csv")

trend_sizes = posts_trends.groupby(["trend_cluster", "trend_label"]).size().reset_index(name="post_count")
trend_sizes = trend_sizes.sort_values("post_count", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(trend_sizes["trend_label"].head(15), trend_sizes["post_count"].head(15))
ax.set_xlabel("Number of Posts")
ax.set_title("Top 15 Detected Trends (by post count)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "trend_detection.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Total trends detected: {posts_trends['trend_cluster'].nunique()}")
print(f"Total user-trend assignments: {len(user_trends)}")
```

**Cell 9 (Markdown):**
```markdown
## 4. Propagation Metrics
```

**Cell 10 (Code):**
```python
with open(RESULTS_DIR / "propagation_metrics.json") as f:
    prop_metrics = json.load(f)

prop_df = pd.DataFrame(prop_metrics)
print(prop_df[["trend_label", "total_users", "growth_rate_12h", "num_subreddits", "mean_post_score"]].to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(prop_df["growth_rate_12h"], prop_df["total_users"])
axes[0, 0].set_xlabel("Growth Rate (users/hr, first 12h)")
axes[0, 0].set_ylabel("Total Users")
axes[0, 0].set_title("Growth Rate vs Total Reach")

axes[0, 1].scatter(prop_df["mean_degree_centrality_early"], prop_df["total_users"])
axes[0, 1].set_xlabel("Mean Degree Centrality (Early Adopters)")
axes[0, 1].set_ylabel("Total Users")
axes[0, 1].set_title("Early Adopter Centrality vs Total Reach")

axes[1, 0].scatter(prop_df["num_subreddits"], prop_df["total_users"])
axes[1, 0].set_xlabel("Number of Subreddits")
axes[1, 0].set_ylabel("Total Users")
axes[1, 0].set_title("Cross-Subreddit Spread vs Total Reach")

axes[1, 1].scatter(prop_df["mean_comment_depth"], prop_df["total_users"])
axes[1, 1].set_xlabel("Mean Comment Depth")
axes[1, 1].set_ylabel("Total Users")
axes[1, 1].set_title("Discussion Depth vs Total Reach")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "propagation_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 11 (Markdown):**
```markdown
## 5. Virality Prediction Results
```

**Cell 12 (Code):**
```python
with open(RESULTS_DIR / "metrics.json") as f:
    model_results = json.load(f)

for model_name, results in model_results.items():
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Accuracy:  {results['mean_accuracy']:.4f}")
    print(f"  Precision: {results['mean_precision']:.4f}")
    print(f"  Recall:    {results['mean_recall']:.4f}")
    print(f"  F1 Score:  {results['mean_f1']:.4f}")

# Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (model_name, results) in zip(axes, model_results.items()):
    imp = results["feature_importance"]
    features = list(imp.keys())
    values = list(imp.values())
    ax.barh(features, values)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 13 (Markdown):**
```markdown
## 6. Conclusions

Key findings from the analysis:
1. **Trend propagation speed** — Which features best predict viral trends?
2. **Network effects** — Do well-connected early adopters accelerate spread?
3. **Cross-subreddit spread** — Do trends that appear in multiple communities grow faster?

These findings support/challenge the hypothesis that social network structure plays a significant role in trend adoption on Reddit.
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/analysis.ipynb
git commit -m "feat: add analysis notebook with visualizations"
```

---

### Task 10: End-to-End Integration & .gitignore

**Files:**
- Create: `.gitignore`
- Verify: full pipeline runs

- [ ] **Step 1: Create .gitignore**

Create `.gitignore`:
```
data/raw/
data/processed/
data/subreddits.json
results/
__pycache__/
*.pyc
.env
*.egg-info/
.venv/
```

- [ ] **Step 2: Verify full test suite passes**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (17 total).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for data and build artifacts"
```

- [ ] **Step 4: Run end-to-end smoke test (requires API keys)**

```bash
# Step 1: Discover subreddits (requires GEMINI_API_KEY)
GEMINI_API_KEY=your_key uv run python -m src.discover

# Step 2: Collect Reddit data
uv run python -m src.collect

# Step 3: Build graph
uv run python -m src.graph

# Step 4: Detect trends
uv run python -m src.trends

# Step 5: Compute propagation metrics
uv run python -m src.analyze

# Step 6: Train models
uv run python -m src.predict
```

- [ ] **Step 5: Final commit with any fixes**

```bash
git add -A
git commit -m "chore: final integration fixes"
```
