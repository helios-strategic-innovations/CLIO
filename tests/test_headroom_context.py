import hashlib
import json
import sys
from pathlib import Path

OPENCLAW_DIR = Path(__file__).parent.parent
if str(OPENCLAW_DIR) not in sys.path:
    sys.path.insert(0, str(OPENCLAW_DIR))

from agent.prompts import headroom_context as hc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack(tmp_path: Path, dropped_chars: int = 0) -> tuple[Path, Path]:
    source = tmp_path / "source.md"
    source.write_text("alpha beta gamma\n", encoding="utf-8")

    pack = tmp_path / "pack"
    page_dir = pack / "source.md"
    page_dir.mkdir(parents=True)
    page = page_dir / "page-001.png"
    page.write_bytes(b"not-a-real-png-but-hashed")
    kept_text = ""
    (pack / "kept-text.ndjson").write_text(kept_text, encoding="utf-8")

    manifest = {
        "packName": "test-pack",
        "packHash": "test-hash",
        "keptTextSha256": hashlib.sha256(kept_text.encode("utf-8")).hexdigest(),
        "sources": [
            {
                "source": str(source),
                "sourceSha256": _sha(source),
                "renderedInputSha256": "unused",
                "keptLines": 0,
                "keptTextChars": 0,
                "sourceChars": source.stat().st_size,
                "droppedChars": dropped_chars,
                "pages": [
                    {
                        "index": 1,
                        "file": "source.md/page-001.png",
                        "width": 100,
                        "height": 50,
                        "bytes": page.stat().st_size,
                        "sha256": _sha(page),
                    }
                ],
            }
        ],
        "totals": {
            "sourceChars": source.stat().st_size,
            "renderedChars": source.stat().st_size,
            "keptLines": 0,
            "keptTextChars": 0,
            "droppedChars": dropped_chars,
            "pixels": 5000,
            "pageCount": 1,
        },
        "estimates": {"reductionOpenAIProse": 0.42},
    }
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack, source


def test_verify_headroom_pack_accepts_zero_drop_pack(tmp_path):
    pack, _source = _write_pack(tmp_path)
    result = hc.verify_headroom_pack(pack, expected_pack_hash="test-hash")
    assert result["ok"] is True
    assert result["manifest"]["totals"]["droppedChars"] == 0


def test_verify_headroom_pack_rejects_dropped_chars(tmp_path):
    pack, _source = _write_pack(tmp_path, dropped_chars=1)
    result = hc.verify_headroom_pack(pack, expected_pack_hash="test-hash")
    assert result["ok"] is False
    assert "dropped chars nonzero" in result["reason"]


def test_verify_headroom_pack_rejects_source_hash_drift(tmp_path):
    pack, source = _write_pack(tmp_path)
    source.write_text("changed\n", encoding="utf-8")
    result = hc.verify_headroom_pack(pack, expected_pack_hash="test-hash")
    assert result["ok"] is False
    assert any("source hash mismatch" in failure for failure in result["failures"])


def test_file_redirect_is_active_only_and_exact_reads_bypass(tmp_path, monkeypatch):
    pack, source = _write_pack(tmp_path)
    monkeypatch.setenv("CLIO_HEADROOM_PACK_HASH", "test-hash")
    monkeypatch.setenv("CLIO_HEADROOM_FILE_READS_ENABLED", "1")
    monkeypatch.setenv("CLIO_HEADROOM_FILE_READS_LOG_ONLY", "0")
    redirect = hc.maybe_headroom_file_redirect(source, "source.md", pack_dir=pack)
    assert redirect is not None
    assert "HEADROOM PACK HIT" in redirect
    assert "page-001.png" in redirect

    exact = hc.maybe_headroom_file_redirect(source, "source.md", exact_requested=True, pack_dir=pack)
    assert exact is None
