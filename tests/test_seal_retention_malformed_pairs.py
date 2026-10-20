"""seal_retention_tester.run_verification_cycle must skip a stored QA pair that
is a bare string (seen on the VPS: AttributeError 'str' object has no attribute
'get', once per daily cycle) instead of crashing the whole cycle.

The tester is built without its __init__ (which touches data/) and every
method that reads/writes files or calls a model is faked.
"""

import asyncio
import threading

from agent.integrations import seal_retention_tester as srt


def test_bare_string_pair_is_skipped_and_cycle_completes(monkeypatch):
    tester = object.__new__(srt.VaultRetentionTester)
    tester._lock = threading.Lock()
    tester._state = {}

    verified: list[tuple[str, str]] = []
    updated: list[tuple[int, int]] = []

    async def _verify(title, content, q, a):
        verified.append((q, a))
        return {"matches": True}

    monkeypatch.setenv("MOLTBOT_SEAL_RETENTION", "1")
    monkeypatch.setattr(tester, "_load_qa_files", lambda: [
        {"entry_id": "e1", "title": "T", "_filepath": "/nonexistent/e1_qa.json",
         "qa_pairs": ["a bare string, not a pair", {"q": "Q1", "a": "A1"}]},
    ], raising=False)
    monkeypatch.setattr(tester, "_load_entry_content", lambda entry_id: ("T", "content"), raising=False)
    monkeypatch.setattr(tester, "_verify_qa_pair", _verify, raising=False)
    monkeypatch.setattr(tester, "_update_qa_file",
                        lambda qa, passed, failed: updated.append((passed, failed)), raising=False)
    monkeypatch.setattr(tester, "_save_state", lambda: None, raising=False)

    result = asyncio.run(tester.run_verification_cycle())

    assert verified == [("Q1", "A1")]       # the dict pair was still tested
    assert updated == [(1, 0)]
    assert not result.get("skipped")
