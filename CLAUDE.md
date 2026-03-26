# Social Contagion of Trends — Data Mining Project

## Project Summary

Academic project analyzing how trends/habits spread through Reddit social networks. Graph-first approach using Reddit's `.json` API for data collection and Gemini API for subreddit discovery.

## Architecture

4-phase pipeline: Discover subreddits (Gemini) -> Collect data (Reddit .json) -> Build graph + detect trends -> Analyze propagation + predict virality

## Key Decisions

- **Package manager**: UV
- **Data source**: Reddit `.json` endpoint (append `.json` to any subreddit URL)
- **Graph library**: NetworkX
- **ML models**: Logistic Regression + Random Forest
- **Trend detection**: TF-IDF + K-Means clustering
- **Scope**: Snapshot analysis (~1 week of data, ~20-30 subreddits)
- **ML target**: Binary virality prediction from first 12h of graph features
- **Evaluation**: Cross-validation (small dataset ~50-100 trend clusters)

## Design Spec

Full spec at: `docs/superpowers/specs/2026-03-25-social-contagion-trends-design.md`

## Commands

```bash
uv run python -m src.discover   # Discover subreddits via Gemini
uv run python -m src.collect    # Collect Reddit data
uv run python -m src.graph      # Build interaction graph
uv run python -m src.trends     # Detect trends via TF-IDF
uv run python -m src.analyze    # Compute propagation metrics
uv run python -m src.predict    # Train virality prediction model
```

## Environment

- Gemini API key needed: set as `GEMINI_API_KEY` env var
