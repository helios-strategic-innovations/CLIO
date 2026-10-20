"""OP-03 (S214): ONEPROVIDER_API_KEY is in the Moltbook secret-name containers."""

from agent.moltbook.client import _FORBIDDEN_ENV_VARS
from agent.moltbook.config import MOLTBOOK_CREDENTIAL_DENYLIST


def test_client_forbidden_env_vars_include_oneprovider():
    assert "ONEPROVIDER_API_KEY" in _FORBIDDEN_ENV_VARS
    assert "ANTHROPIC_API_KEY" in _FORBIDDEN_ENV_VARS


def test_config_credential_denylist_includes_oneprovider():
    assert "ONEPROVIDER_API_KEY" in MOLTBOOK_CREDENTIAL_DENYLIST
    assert "ANTHROPIC_API_KEY" in MOLTBOOK_CREDENTIAL_DENYLIST
