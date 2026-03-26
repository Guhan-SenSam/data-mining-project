# Social Contagion of Trends

Academic data mining project analyzing how habits and trends spread through Reddit communities. Uses a graph-based approach: scrape posts/comments → build a user interaction graph → detect trending topics → measure propagation speed → predict virality with ML.

## Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd data-mining-project
uv sync
```

**2. Set your Gemini API key**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

To persist it, add the line above to your `~/.bashrc` or `~/.zshrc`.

## Running the Pipeline

Run each phase in order:

```bash
# 1. Discover relevant subreddits via Gemini (outputs data/subreddits.json)
uv run python -m src.discover

# 2. Scrape posts and comments from discovered subreddits (outputs data/raw/ and data/processed/)
uv run python -m src.collect

# 3. Build the user interaction graph (outputs data/processed/graph.gpickle)
uv run python -m src.graph

# 4. Detect trends via TF-IDF + K-Means clustering
uv run python -m src.trends

# 5. Compute propagation metrics per trend cluster
uv run python -m src.analyze

# 6. Train virality prediction models (Logistic Regression + Random Forest)
uv run python -m src.predict
```

Each step caches its output, so you can re-run individual phases without restarting from scratch.

## Project Structure

```
data-mining-project/
├── src/
│   ├── discover.py     # Gemini API subreddit discovery
│   ├── collect.py      # Reddit web scraper
│   ├── graph.py        # NetworkX interaction graph construction
│   ├── trends.py       # TF-IDF + K-Means trend detection
│   ├── analyze.py      # Propagation metrics computation
│   ├── predict.py      # ML virality prediction
│   └── utils.py        # Rate limiting, config, shared helpers
├── notebooks/
│   └── analysis.ipynb  # Interactive analysis and visualizations
├── data/
│   ├── subreddits.json # Cached subreddit list from Gemini
│   ├── raw/            # Raw scraped responses
│   └── processed/      # Cleaned CSVs (posts.csv, comments.csv)
├── results/
│   ├── figures/        # Saved plots
│   └── metrics.json    # Model evaluation results
└── pyproject.toml
```

## Jupyter Notebook

To explore results interactively:

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

## Running Tests

```bash
uv run pytest
```
