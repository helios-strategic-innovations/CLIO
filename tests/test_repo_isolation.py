"""Tests never read the LIVE advisory state files under data/intelligence/.

On the VPS those files exist and are fresh (the flow collector rewrites
selfflow_state.json every 15 s even while idle); tests that fell through to
the default path read live data (13 tests/dpof failures, 2026-09-25).
tests/conftest.py points every such path at a per-test temp dir."""

import os

from agent.flowcollector import state as flowcollector_state


def test_live_state_paths_are_redirected_to_temp():
    for key in ("CLIO_SELFFLOW_FILE", "CLIO_SELFFLOW_COMPARE_FILE", "CLIO_VOL_SURFACE_FILE",
                "CLIO_ECON_SURPRISE_FILE", "CLIO_NEWSFLASH_FILE"):
        path = os.environ.get(key, "")
        assert path and "data/intelligence" not in path, key
    assert flowcollector_state.state_path() != flowcollector_state.DEFAULT_STATE_FILE
