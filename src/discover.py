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
