import asyncio
import json
import sys
import types
import importlib.util
from pathlib import Path
from types import SimpleNamespace


# The checkout that CONTAINS this test. (Not "a sibling directory named openclaw": in any git worktree that resolved to
# the live serving tree, so the test silently checked the wrong code -- and failed outright in a checkout under /tmp.)
OPENCLAW = Path(__file__).resolve().parents[1]
SYNTHETIC_PASSPHRASE = "ASTER_SYNTHETIC_OVERRIDE_PHRASE"
SYNTHETIC_PIN = "864209"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_voice_override_stack(monkeypatch, tmp_path):
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("OVERRIDE_PASSPHRASE", SYNTHETIC_PASSPHRASE)
    # B2a (2026-09-24): dispatch_directive() now no-ops unless the Hydrinia bridge
    # is explicitly re-enabled (Hydrinia is permanently off). This suite tests the
    # PIN-redaction / directive-dispatch plumbing itself, not the bridge gate, so
    # keep it exercising the real write path.
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")

    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX = 2
    fcntl.LOCK_UN = 8
    fcntl.flock = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "fcntl", fcntl)

    agent = types.ModuleType("agent")
    agent.__path__ = [str(OPENCLAW / "agent")]
    monkeypatch.setitem(sys.modules, "agent", agent)

    config_mod = types.ModuleType("agent.config")
    config_mod.config = SimpleNamespace(
        adaptive_thinking=False,
        anthropic_api_key="test-only",
        max_tokens=128,
        override_directives_enabled=True,
        override_voice_pin=SYNTHETIC_PIN,
        server_compaction=False,
        telegram_authorized_users=[],
    )

    class Config:
        @staticmethod
        def from_env():
            return SimpleNamespace(tradingbot_dir=str(tmp_path / "tradingbot"))

    config_mod.Config = Config
    monkeypatch.setitem(sys.modules, "agent.config", config_mod)

    anthropic = types.ModuleType("anthropic")
    anthropic.APIError = type("APIError", (Exception,), {})
    anthropic.BadRequestError = type("BadRequestError", (anthropic.APIError,), {})
    class Anthropic:
        def __init__(self, *_args, **_kwargs):
            self.args = _args
            self.kwargs = _kwargs

    anthropic.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)

    security = types.ModuleType("agent.security")
    security.CONFIRMATION_REQUIRED = set()
    security.sanitize_user_input = lambda value: value
    security.validate_tool_call = lambda *_args, **_kwargs: (True, "")
    monkeypatch.setitem(sys.modules, "agent.security", security)

    prompts = types.ModuleType("agent.prompts")
    prompts.__path__ = []
    system_prompt = types.ModuleType("agent.prompts.system_prompt")
    system_prompt.get_system_prompt = lambda *_args, **_kwargs: []
    system_prompt.get_model_for_task = lambda *_args, **_kwargs: "test-model"
    monkeypatch.setitem(sys.modules, "agent.prompts", prompts)
    monkeypatch.setitem(sys.modules, "agent.prompts.system_prompt", system_prompt)

    tools = types.ModuleType("agent.tools")
    tools.__path__ = []
    definitions = types.ModuleType("agent.tools.definitions")
    definitions.get_all_tools = lambda: []
    executor = types.ModuleType("agent.tools.executor")

    async def execute_tool(*_args, **_kwargs):
        return "not-used"

    executor.execute_tool = execute_tool
    monkeypatch.setitem(sys.modules, "agent.tools", tools)
    monkeypatch.setitem(sys.modules, "agent.tools.definitions", definitions)
    monkeypatch.setitem(sys.modules, "agent.tools.executor", executor)

    memory = types.ModuleType("agent.memory")
    memory.__path__ = []
    persistence = types.ModuleType("agent.memory.persistence")

    class PersistentConversationManager:
        async def get_history(self, _conversation_id):
            return []

        async def add_message(self, *_args, **_kwargs):
            return None

    persistence.PersistentConversationManager = PersistentConversationManager
    monkeypatch.setitem(sys.modules, "agent.memory", memory)
    monkeypatch.setitem(sys.modules, "agent.memory.persistence", persistence)

    safety = types.ModuleType("agent.safety")
    safety.__path__ = []
    audit_log = types.ModuleType("agent.safety.audit_log")
    audit_events = []

    def audit_security_event(event_type, message, details=None):
        audit_events.append(
            {"event_type": event_type, "message": message, "details": details or {}}
        )

    audit_log.audit_security_event = audit_security_event
    monkeypatch.setitem(sys.modules, "agent.safety", safety)
    monkeypatch.setitem(sys.modules, "agent.safety.audit_log", audit_log)

    integrations = types.ModuleType("agent.integrations")
    integrations.__path__ = [str(OPENCLAW / "agent" / "integrations")]
    monkeypatch.setitem(sys.modules, "agent.integrations", integrations)

    override = _load_module(
        "agent.override_directive",
        OPENCLAW / "agent" / "override_directive.py",
    )
    override.OVERRIDE_LOG_PATH = str(tmp_path / "override_directive.jsonl")

    delivery = _load_module(
        "agent.integrations.directive_delivery",
        OPENCLAW / "agent" / "integrations" / "directive_delivery.py",
    )
    delivery.DIRECTIVES_DIR = str(tmp_path / "tradingbot" / "moltbot_knowledge" / "directives")
    delivery.DIRECTIVES_ARCHIVE_DIR = str(Path(delivery.DIRECTIVES_DIR) / "archive")
    delivery.DIRECTIVE_RESPONSES_DIR = str(
        tmp_path / "tradingbot" / "moltbot_knowledge" / "directive_responses"
    )
    delivery.DIRECTIVES_LOG = str(tmp_path / "directives.jsonl")

    core = _load_module("agent.core", OPENCLAW / "agent" / "core.py")
    return core, override, delivery, audit_events


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_voice_pin_is_stripped_before_override_log_and_audit_preview(monkeypatch, tmp_path):
    core, _override, delivery, audit_events = _load_voice_override_stack(monkeypatch, tmp_path)
    spoken = f"{SYNTHETIC_PASSPHRASE} PIN {SYNTHETIC_PIN} Consider SPY only"

    response = asyncio.run(
        core.agent_respond(
            user_message=spoken,
            conversation_id="voice_positive",
            source="twilio_voice",
            user_id="+15551234567",
            voice_mode=True,
        )
    )

    assert "PIN" not in response
    assert SYNTHETIC_PIN not in response

    override_entries = _jsonl(tmp_path / "override_directive.jsonl")
    directive_entries = _jsonl(Path(delivery.DIRECTIVES_LOG))

    assert len(override_entries) == 1
    assert len(directive_entries) == 1

    persisted = json.dumps(
        {"override": override_entries, "audit": audit_events, "directive": directive_entries},
        sort_keys=True,
    )
    assert SYNTHETIC_PIN not in persisted
    assert f"PIN {SYNTHETIC_PIN}" not in persisted


def test_valid_voice_pin_dispatches_clean_non_pin_command(monkeypatch, tmp_path):
    core, _override, delivery, _audit_events = _load_voice_override_stack(monkeypatch, tmp_path)
    spoken = f"{SYNTHETIC_PASSPHRASE} PIN {SYNTHETIC_PIN} Consider SPY only"

    response = asyncio.run(
        core.agent_respond(
            user_message=spoken,
            conversation_id="voice_positive_control",
            source="twilio_voice",
            user_id="+15551234567",
            voice_mode=True,
        )
    )

    directive_entries = _jsonl(Path(delivery.DIRECTIVES_LOG))
    directive_files = list(Path(delivery.DIRECTIVES_DIR).glob("DIR-*.json"))

    assert "Directive dispatched." in response
    assert len(directive_entries) == 1
    assert len(directive_files) == 1
    assert directive_entries[0]["command"] == "Consider SPY only"
    assert json.loads(directive_files[0].read_text(encoding="utf-8"))["command"] == "Consider SPY only"


def test_voice_pin_candidate_is_removed_before_directive_dispatch(monkeypatch, tmp_path):
    core, _override, delivery, _audit_events = _load_voice_override_stack(monkeypatch, tmp_path)
    spoken = f"{SYNTHETIC_PASSPHRASE} PIN {SYNTHETIC_PIN} Consider SPY only"
    dispatch_observation = {}

    async def capture_dispatch_event(override_event, _raw_message):
        dispatch_observation["pin_attr_present"] = hasattr(
            override_event, "_voice_pin_candidate"
        )
        dispatch_observation["event_dict"] = dict(vars(override_event))
        dispatch_observation["command_text"] = override_event.command_text
        return "Directive dispatched. test capture"

    monkeypatch.setattr(delivery, "dispatch_directive", capture_dispatch_event)

    response = asyncio.run(
        core.agent_respond(
            user_message=spoken,
            conversation_id="voice_ephemeral",
            source="twilio_voice",
            user_id="+15551234567",
            voice_mode=True,
        )
    )

    assert response == "Directive dispatched. test capture"
    assert dispatch_observation["command_text"] == "Consider SPY only"
    assert dispatch_observation["pin_attr_present"] is False
    assert "_voice_pin_candidate" not in dispatch_observation["event_dict"]


def test_failed_voice_pin_increments_lockout_without_persisting_attempted_value(
    monkeypatch, tmp_path
):
    core, override, delivery, audit_events = _load_voice_override_stack(monkeypatch, tmp_path)
    attempted_pin = "864210"
    spoken = f"{SYNTHETIC_PASSPHRASE} PIN {attempted_pin} Execute SPY now"

    response = asyncio.run(
        core.agent_respond(
            user_message=spoken,
            conversation_id="voice_failed",
            source="twilio_voice",
            user_id="+15557654321",
            voice_mode=True,
        )
    )

    assert response == "Voice override requires valid PIN. Command not executed."
    assert not list(Path(delivery.DIRECTIVES_DIR).glob("DIR-*.json"))
    assert not Path(delivery.DIRECTIVES_LOG).exists()

    pin_state = override._pin_attempts[override._pin_key("+15557654321", "voice_failed")]
    assert pin_state["fails"] == 1

    persisted = json.dumps(
        {"override": _jsonl(tmp_path / "override_directive.jsonl"), "audit": audit_events},
        sort_keys=True,
    )
    assert attempted_pin not in persisted
    assert f"PIN {attempted_pin}" not in persisted
