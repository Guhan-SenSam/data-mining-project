# Social Contagion of Trends: Design Spec

## Overview

Academic project studying how trends and habits spread through social networks on Reddit. Uses a graph-first approach: collect posts and comments from habit-related subreddits, build a user interaction graph, detect trending topics, measure propagation speed, and train an ML model to predict trend virality from early-spread graph features.

## Scope

- **Snapshot analysis**: last ~1 week of data from ~20-30 subreddits
- **Focus**: trend detection + propagation speed measurement
- **ML target**: predict whether a trend will go viral based on first 12h of graph features
- **Package manager**: UV
- **Language**: Python

## Pipeline Phases

### Phase 1: Subreddit Discovery (Gemini API)

- Prompt Gemini to return ~20-30 subreddits across habit categories (fitness, study, productivity, diet, sleep, meditation, etc.)
- Output: `data/subreddits.json` — a cached JSON list of subreddit names
- One-shot call, cached for reruns

### Phase 2: Data Collection (Reddit `.json` API)

- For each subreddit, fetch `/r/{subreddit}/new.json` and `/r/{subreddit}/hot.json` (up to ~100 posts each, paginated via `after` param)
- For each post, fetch `/r/{subreddit}/comments/{post_id}.json` for full comment tree
- Rate limiting: 1-2s delay between requests, custom `User-Agent` header
- Store raw JSON in `data/raw/` for reproducibility
- Parse into structured CSVs:

**posts.csv**: post_id, subreddit, author, title, selftext, score, num_comments, created_utc

**comments.csv**: comment_id, post_id, parent_id, author, body, score, created_utc

### Phase 3: Graph Construction & Trend Detection

**User Interaction Graph (NetworkX):**
- Nodes = unique usernames (excluding `[deleted]`, `AutoModerator`, bots)
- Directed edges: commenter -> post author, commenter -> parent comment author
- Edge weights = number of interactions between two users
- Edge attributes: subreddit, timestamp, post_id

**Trend Detection (TF-IDF + K-Means):**
- Combine post titles + selftext into documents
- TF-IDF vectorization -> K-Means clustering to identify topic clusters (trends)
- Label each cluster by top-3 TF-IDF terms (e.g., "creatine, supplement, dose")
- Assign each post and its commenters to a trend cluster
- User "adopts" a trend = they post or comment in that topic cluster

### Phase 4: Propagation Analysis & ML Prediction

**Propagation Metrics per trend:**
- Time of first appearance (earliest post timestamp)
- Adoption curve: cumulative unique users over time
- Propagation speed: users-per-hour in first 24h
- Network spread: avg degree centrality of early adopters (first 25%) vs late adopters
- Cross-subreddit spread: number of distinct subreddits the trend reaches

**Virality Prediction Model:**
- Unit of prediction: one trend cluster
- Target: binary — viral (above-median total reach) vs not-viral
- Features extracted from first 12h of trend life:
  - Number of unique users engaged
  - Growth rate (users/hour)
  - Mean degree centrality of early adopters
  - Number of subreddits reached
  - Mean post score
  - Comment depth (avg reply chain length)
- Models: Logistic Regression (interpretable baseline) + Random Forest
- Evaluation: cross-validation (dataset ~50-100 clusters), accuracy, precision, recall, F1, feature importance
- Note: small dataset necessitates cross-validation over train/test split

## Project Structure

```
data-mining-project/
├── CLAUDE.md                  # Project plan & notes
├── pyproject.toml             # UV project config
├── src/
│   ├── __init__.py
│   ├── discover.py            # Gemini API subreddit discovery
│   ├── collect.py             # Reddit data collection
│   ├── graph.py               # Graph construction
│   ├── trends.py              # TF-IDF trend detection
│   ├── analyze.py             # Propagation metrics
│   ├── predict.py             # ML virality prediction
│   └── utils.py               # Rate limiting, config, helpers
├── notebooks/
│   └── analysis.ipynb         # Main analysis notebook with visualizations
├── data/
│   ├── subreddits.json        # Gemini-generated subreddit list
│   ├── raw/                   # Raw JSON responses
│   └── processed/             # Cleaned CSVs
└── results/
    ├── figures/               # Saved plots
    └── metrics.json           # Model performance
```

## Dependencies

- `requests` — HTTP calls to Reddit
- `networkx` — graph construction and analysis
- `scikit-learn` — TF-IDF, K-Means, Logistic Regression, Random Forest
- `pandas` — data wrangling
- `matplotlib` + `seaborn` — visualization
- `google-generativeai` — Gemini API for subreddit discovery
- `jupyter` — notebook for analysis

## Approach Rationale

Graph-first was chosen over time-series-only or hybrid because:
1. Directly models social contagion (how trends spread *through people*)
2. Graph metrics (centrality, clustering) provide academic depth
3. Keeps scope manageable for snapshot analysis
4. Temporal features (timestamps) are still captured on edges without a separate time-series pipeline
