import asyncio
import pathlib
import sys
import tempfile
import time
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_main_leaves_sigterm_to_ptb():
    text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "signal.signal(signal.SIGTERM" not in text
    assert "_sigterm_handler" not in text


def test_shutdown_observability_markers_present():
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    telegram_text = (ROOT / "agent/integrations/telegram_bot.py").read_text(encoding="utf-8")
    scheduled_text = (ROOT / "agent/integrations/scheduled_tasks.py").read_text(encoding="utf-8")

    assert "_log_shutdown_threads" in main_text
    assert "Shutdown thread dump:" in main_text
    assert "CLIO shutdown stage ENTER: post_shutdown" in telegram_text
    assert "CLIO shutdown stage ENTER: shutdown_background_tasks" in telegram_text
    assert "CLIO shutdown stage ENTER: memory_flush" in telegram_text
    assert "CLIO shutdown stage ENTER: shutdown_background_tasks.body" in scheduled_text


def test_shutdown_closes_persistence_and_executor_survivors():
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    telegram_text = (ROOT / "agent/integrations/telegram_bot.py").read_text(encoding="utf-8")
    web_search_text = (ROOT / "agent/integrations/web_search.py").read_text(encoding="utf-8")
    memory_text = (ROOT / "agent/memory/memory_engine.py").read_text(encoding="utf-8")

    assert "def flush(self)" in memory_text
    assert "shutdown_cleanup_completed" in telegram_text
    assert "conversation_manager.close()" in telegram_text
    assert "CLIO shutdown stage ENTER: conversation_persistence_close" in telegram_text
    assert "shutdown_search_executors" in telegram_text
    assert "CLIO shutdown stage ENTER: web_search_executor_shutdown" in telegram_text
    assert "shutdown_default_executor" in telegram_text
    assert "CLIO shutdown stage ENTER: default_executor_shutdown" in telegram_text
    assert "_final_shutdown_guard" in main_text
    assert "os._exit(0)" in main_text
    assert "_release_pid_lock()" in main_text
    assert "DDGS" in web_search_text
    assert "cancel_futures=True" in web_search_text


def test_memory_engine_flush_delegates_to_export_hot_context():
    from agent.memory.memory_engine import MemoryEngine

    called = []
    original = MemoryEngine.export_hot_context
    try:
        MemoryEngine.export_hot_context = lambda self: called.append(True)
        engine = object.__new__(MemoryEngine)
        MemoryEngine.flush(engine)
    finally:
        MemoryEngine.export_hot_context = original

    assert called == [True]


def test_persistent_conversation_manager_close_releases_connection():
    from agent.memory import persistence

    old_db_path = persistence.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        persistence.DB_PATH = pathlib.Path(tmp) / "conversations.db"
        manager = persistence.PersistentConversationManager(max_history=3)

        async def scenario():
            await manager.add_message("shutdown-test", "user", "hello")
            assert manager._db is not None
            await manager.close()
            assert manager._db is None

        try:
            asyncio.run(scenario())
        finally:
            persistence.DB_PATH = old_db_path


def test_shutdown_search_executors_closes_cached_ddgs_executor():
    fake_ddgs_mod = types.ModuleType("ddgs")

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def shutdown(self, **kwargs):
            self.calls.append(kwargs)

    class FakeDDGS:
        _executor = FakeExecutor()

    fake_ddgs_mod.DDGS = FakeDDGS
    original = sys.modules.get("ddgs")
    sys.modules["ddgs"] = fake_ddgs_mod
    try:
        from agent.integrations.web_search import shutdown_search_executors

        assert shutdown_search_executors(wait=False) is True
        assert FakeDDGS._executor is None
    finally:
        if original is None:
            sys.modules.pop("ddgs", None)
        else:
            sys.modules["ddgs"] = original


def test_price_stream_stop_waits_for_runner_future():
    config_mod = types.ModuleType("agent.config")
    config_mod.config = types.SimpleNamespace()
    sys.modules.setdefault("agent.config", config_mod)

    from agent.integrations.price_stream import PriceStreamManager

    class FakeStream:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    async def scenario():
        mgr = PriceStreamManager()
        stream = FakeStream()
        mgr._running = True
        mgr._stock_stream = stream

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, time.sleep, 0.05)
        mgr._stock_run_future = future

        await mgr.stop()
        assert stream.stopped is True
        assert future.done() is True
        assert mgr._stock_run_future is None

    asyncio.run(scenario())
