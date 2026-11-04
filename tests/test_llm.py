"""The model switch and the lenient JSON reader used for open-weights output."""

import pytest

from vericell import llm


def test_first_json_tolerates_fences_and_run_on_text():
    assert llm._first_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._first_json('{"a": 1} and then the model kept talking {"b": 2}') == {"a": 1}
    with pytest.raises(ValueError):
        llm._first_json("no json here")


def test_backend_switch(monkeypatch):
    monkeypatch.delenv("VERICELL_LLM", raising=False)
    llm.use(None)
    assert llm.backend() == "meta"
    llm.use("open")
    assert llm.backend() == "open"
    llm.use("nonsense")
    assert llm.backend() == "meta"
