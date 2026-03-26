"""Discover trending habit-related subreddits using Gemini API."""

import json
import os

from google import genai

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

FALLBACK_SUBREDDITS = [
    "Fitness", "running", "bodyweightfitness", "gym", "loseit",
    "EatCheapAndHealthy", "MealPrepSunday", "GetStudying", "studytips",
    "productivity", "getdisciplined", "selfimprovement", "DecidingToBeBetter",
    "Meditation", "mindfulness", "sleep", "writing", "drawing",
    "theXeffect", "NonZeroDay", "HabitExchange", "dailygoals",
    "C25K", "flexibility", "yoga", "intermittentfasting"
]


def discover_subreddits() -> list[str]:
    """Return list of subreddit names, using cache if available."""
    if SUBREDDITS_PATH.exists():
        return json.loads(SUBREDDITS_PATH.read_text())

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY set. Using fallback subreddit list.")
        _save(FALLBACK_SUBREDDITS)
        return FALLBACK_SUBREDDITS

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PROMPT,
        )
        subreddits = json.loads(response.text)
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        print("Using fallback subreddit list.")
        subreddits = FALLBACK_SUBREDDITS

    _save(subreddits)
    return subreddits


def _save(subreddits: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUBREDDITS_PATH.write_text(json.dumps(subreddits, indent=2))


if __name__ == "__main__":
    subs = discover_subreddits()
    print(f"Discovered {len(subs)} subreddits: {subs}")
