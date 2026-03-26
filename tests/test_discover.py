import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.discover import discover_subreddits, SUBREDDITS_PATH


def test_discover_subreddits_returns_list(tmp_path):
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        "Fitness", "GetStudying", "productivity", "loseit",
        "Meditation", "getdisciplined"
    ])
    mock_model.generate_content.return_value = mock_response

    # Use a non-existent cache path so the function proceeds to call the API
    missing_cache = tmp_path / "subreddits.json"

    with patch("src.discover.genai") as mock_genai, \
         patch("src.discover.SUBREDDITS_PATH", missing_cache), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
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
