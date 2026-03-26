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
RATE_LIMIT_SECONDS = 1.0


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
