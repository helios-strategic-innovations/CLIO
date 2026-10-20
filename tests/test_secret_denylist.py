"""A-3 (P0-17): secret-file denylist coverage.

Targets ``agent/security.py`` ``_is_blocked_secret_file`` and its wiring into
``agent/tools/file_tools.py`` ``_resolve_path``. Secret files (``.env`` /
``.env.*`` / private keys / certs) must be blocked; ordinary files must pass;
a path-traversal that resolves onto a secret file inside an allowed root must be
blocked (defense-in-depth), and a traversal that escapes the allowed roots must
be blocked as out-of-bounds.

``agent.security`` guards its optional ``definitions`` import, so this collects
without third-party deps installed.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.security import _is_blocked_secret_file  # noqa: E402
from agent.tools.file_tools import _resolve_path  # noqa: E402

_ALLOWED_ROOT = "/home/agent/openclaw/"


@pytest.mark.parametrize(
    "name",
    [".env", ".env.production", ".env.soak", "key.pem", "server.crt", "x.p12"],
)
def test_secret_files_are_blocked(name):
    # Blocked whether given a full resolved path or a bare basename.
    assert _is_blocked_secret_file(_ALLOWED_ROOT + name) is True
    assert _is_blocked_secret_file(name) is True


@pytest.mark.parametrize("name", ["notes.md", "data.json"])
def test_ordinary_files_are_allowed(name):
    assert _is_blocked_secret_file(_ALLOWED_ROOT + name) is False
    assert _is_blocked_secret_file(name) is False


def test_resolve_path_blocks_secret_in_allowed_root():
    ok, resolved, err = _resolve_path("/home/agent/openclaw/.env")
    assert ok is False
    assert resolved == ""
    assert "secret" in err.lower()


def test_resolve_path_blocks_traversal_onto_secret():
    # "agent/../.env" realpath-normalises to /home/agent/openclaw/.env — inside an
    # allowed root but a secret file, so the denylist must still catch it.
    ok, resolved, err = _resolve_path("agent/../.env")
    assert ok is False
    assert "secret" in err.lower()


def test_resolve_path_blocks_traversal_outside_roots():
    ok, resolved, err = _resolve_path("/home/agent/openclaw/../../../etc/passwd")
    assert ok is False
    assert "outside allowed" in err.lower()


def test_resolve_path_allows_ordinary_file():
    ok, resolved, err = _resolve_path("/home/agent/openclaw/notes.md")
    assert ok is True
    assert resolved == "/home/agent/openclaw/notes.md"
    assert err == ""
