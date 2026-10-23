import json

import pytest

from app.services.llm_json import extract_json


def test_extract_json_plain():
    assert extract_json('{"score": 1}') == {"score": 1}


def test_extract_json_fenced_with_json_tag():
    assert extract_json('```json\n{"score": 1}\n```') == {"score": 1}


def test_extract_json_fenced_without_language_tag():
    assert extract_json('```\n{"score": 1}\n```') == {"score": 1}


def test_extract_json_malformed_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json("not json at all")
