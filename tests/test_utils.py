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
