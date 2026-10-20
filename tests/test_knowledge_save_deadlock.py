#!/usr/bin/env python3
"""S205 E1/E5 regression tests — knowledge-save deadlock + commodity price gate.

Run (no pytest needed):  ./venv/bin/python3 tests/test_knowledge_save_deadlock.py

All file I/O is redirected into a throwaway temp dir — production knowledge
paths are never touched.

E1 — the S204 Class-A repro: a natural_gas save with $-prices in content used
to self-deadlock at the price-sanity gate (second FD requesting LOCK_SH on
knowledge_index.json.lock while the same thread held LOCK_EX from the create
path — anti-pattern #20). Each save runs in a daemon worker thread with a
bounded join: if the second flock is ever reintroduced, the test FAILS within
JOIN_TIMEOUT seconds instead of hanging the runner.

E5 — the gate only exact-matched 'natural_gas'; crude_oil/metals/agriculture
never entered it. One case per commodity proves the save lands in the index
AND carries price_divergence_warning when prices diverge >25% from a seeded
prior — i.e. the gate actually evaluated, not just that the save returned.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

JOIN_TIMEOUT = 20.0

# Stub the fire-and-forget integrations BEFORE importing knowledge_tools so the
# save path never reaches contradiction scanning or SEAL enrichment (both lazy
# imports inside handle_save_knowledge).
_cd_stub = types.ModuleType("agent.integrations.contradiction_detector")
_cd_stub.check_new_entry_contradictions = lambda entry: []
sys.modules["agent.integrations.contradiction_detector"] = _cd_stub

_seal_stub = types.ModuleType("agent.tools.seal_knowledge_enhancer")
_seal_stub.SEAL_ENABLED = lambda: False
_seal_stub.SEAL_LOG_ONLY = lambda: True
_seal_stub.get_implication_engine = lambda: None
sys.modules["agent.tools.seal_knowledge_enhancer"] = _seal_stub

from agent.tools import knowledge_tools as kt  # noqa: E402

CATEGORY = "trading_market_analysis"
REL_DIR = kt.CATEGORY_DIRS[CATEGORY]


def patch_paths(mod, base: str) -> None:
    """Point every path constant the save path touches into the temp dir."""
    mod.KNOWLEDGE_BASE_DIR = base
    mod.KNOWLEDGE_INDEX_FILE = os.path.join(base, "knowledge_index.json")
    mod.KNOWLEDGE_ARCHIVE_INDEX_FILE = os.path.join(base, "knowledge_index_archive.json")
    mod._INDEX_LOCK_FILE = mod.KNOWLEDGE_INDEX_FILE + ".lock"
    mod.TRADINGBOT_MIRROR_DIR = os.path.join(base, "mirror")
    mod.URGENT_MIRROR_DIR = os.path.join(base, "mirror", "urgent")
    mod.CURRENT_PRIORS_PATH = os.path.join(base, "current_priors.md")


def run_save_bounded(mod, payload: dict, timeout: float = JOIN_TIMEOUT):
    """Run handle_save_knowledge in a daemon thread with its own event loop.

    Returns the result string, raises the worker's exception, or returns None
    when the worker is still alive after `timeout` — the deadlock signature.
    Daemon thread: a deadlocked worker can't block interpreter exit.
    """
    box: dict = {}

    def worker():
        loop = asyncio.new_event_loop()
        try:
            box["ret"] = loop.run_until_complete(mod.handle_save_knowledge(payload))
        except Exception as e:  # surfaced to the caller below
            box["exc"] = e
        finally:
            loop.close()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    if "exc" in box:
        raise box["exc"]
    return box["ret"]


def seed_prior(mod, base: str, commodity: str, content: str) -> None:
    """Plant one indexed prior entry for `commodity` with $-prices in content."""
    os.makedirs(os.path.join(base, REL_DIR), exist_ok=True)
    eid = f"prior_{commodity}"
    rel_path = os.path.join(REL_DIR, f"{eid}.json")
    entry = {
        "id": eid,
        "title": f"Prior {commodity} supply analysis",
        "category": CATEGORY,
        "content": content,
        "tags": ["commodity_research", commodity, "2026-06-10"],
        "created": "2026-06-10T08:00:00-04:00",
        "source": "test_seed",
    }
    with open(os.path.join(base, rel_path), "w") as f:
        json.dump(entry, f)
    index = {"version": 1, "entries": [{
        "id": eid, "title": entry["title"], "category": CATEGORY,
        "tags": entry["tags"], "created": entry["created"],
        "file_path": rel_path, "source": "test_seed", "quality_score": 6.0,
    }]}
    with open(mod.KNOWLEDGE_INDEX_FILE, "w") as f:
        json.dump(index, f)


def load_index_entries(mod) -> list:
    with open(mod.KNOWLEDGE_INDEX_FILE) as f:
        return json.load(f).get("entries", [])


def test_e1_ng_save_completes_without_deadlock() -> str:
    """The mechanical S204 repro: NG-with-prices save, no prior, bounded join."""
    with tempfile.TemporaryDirectory(prefix="clio_kt_e1_") as base:
        patch_paths(kt, base)
        ret = run_save_bounded(kt, {
            "title": "Natural Gas Supply Balance regression probe",
            "category": CATEGORY,
            "content": "Henry Hub spot at $3.45 with storage at $3.60 forward.",
            "tags": ["commodity_research", "natural_gas", "2026-06-12"],
            "source": "test",
        })
        assert ret is not None, (
            f"DEADLOCK: save did not complete within {JOIN_TIMEOUT}s — "
            "second-FD flock on the index lock file has been reintroduced "
            "(anti-pattern #20, S204 Class A)"
        )
        assert ret.startswith("Saved:"), f"unexpected result: {ret!r}"
        titles = [e.get("title") for e in load_index_entries(kt)]
        assert "Natural Gas Supply Balance regression probe" in titles, "entry missing from index"
    return "save completed + indexed within bounded time"


def test_e5_per_commodity_gate_fires() -> str:
    """Each commodity_type tag must enter the price gate AND land in the index."""
    results = []
    for commodity in ("natural_gas", "crude_oil", "metals", "agriculture"):
        with tempfile.TemporaryDirectory(prefix=f"clio_kt_e5_{commodity}_") as base:
            patch_paths(kt, base)
            seed_prior(kt, base, commodity, "Benchmark price printed at $100.00 last week.")
            title = f"{commodity} divergence probe Jun 12"
            ret = run_save_bounded(kt, {
                "title": title,
                "category": CATEGORY,
                "content": "Benchmark now trading at $200.00 with $210.00 forward strip.",
                "tags": ["commodity_research", commodity, "2026-06-12"],
                "source": "test",
            })
            assert ret is not None, f"DEADLOCK on {commodity} save (bounded join expired)"
            assert ret.startswith("Saved:"), f"{commodity}: unexpected result {ret!r}"
            new = [e for e in load_index_entries(kt) if e.get("title") == title]
            assert new, f"{commodity}: saved entry missing from index"
            assert "price_divergence_warning" in (new[0].get("tags") or []), (
                f"{commodity}: gate did NOT fire — price_divergence_warning tag absent "
                "(dead-gate regression: tag no longer matches _commodity_keywords)"
            )
            entry_file = os.path.join(base, new[0].get("file_path", ""))
            assert os.path.isfile(entry_file), f"{commodity}: entry file missing on disk"
            results.append(commodity)
    return f"gate fired + save landed for: {', '.join(results)}"


def main() -> int:
    tests = [
        ("E1 deadlock regression (NG-with-prices, bounded)", test_e1_ng_save_completes_without_deadlock),
        ("E5 per-commodity gate + save", test_e5_per_commodity_gate_fires),
    ]
    failed = 0
    for name, fn in tests:
        try:
            detail = fn()
            print(f"PASS  {name} — {detail}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name} — {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name} — {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
