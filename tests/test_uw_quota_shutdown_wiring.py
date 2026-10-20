from pathlib import Path


def test_clio_shutdown_explicitly_closes_quota_worker_with_bound():
    source = (
        Path(__file__).parents[1] / "agent" / "integrations" / "scheduled_tasks.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("async def shutdown_background_tasks"):source.index("async def _cleanup_bridge_files")]
    assert "close_all_uw_quota_clients" in block
    assert "CLIO_UW_QUOTA_SHUTDOWN_TIMEOUT_S" in block
    assert "run_in_executor" in block
