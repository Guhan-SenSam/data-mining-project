import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.discover import discover_subreddits, SUBREDDITS_PATH, FALLBACK_SUBREDDITS


def test_discover_subreddits_returns_list(tmp_path):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        "Fitness", "GetStudying", "productivity", "loseit",
        "Meditation", "getdisciplined"
    ])
    mock_client.models.generate_content.return_value = mock_response

    missing_cache = tmp_path / "subreddits.json"

    with patch("src.discover.genai.Client", return_value=mock_client), \
         patch("src.discover.SUBREDDITS_PATH", missing_cache), \
         patch("src.discover.DATA_DIR", tmp_path), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
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


def test_discover_fallback_without_api_key(tmp_path):
    missing_cache = tmp_path / "subreddits.json"

    with patch("src.discover.SUBREDDITS_PATH", missing_cache), \
         patch("src.discover.DATA_DIR", tmp_path), \
         patch.dict("os.environ", {}, clear=True):
        result = discover_subreddits()

    assert result == FALLBACK_SUBREDDITS
