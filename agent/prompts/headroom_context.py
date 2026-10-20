"""
Headroom printer pack verification and CLIO prompt/file-tool adapters.

The image pages are lossy prose context. Original text remains canonical, and
exact-risk text is only available through the kept-text sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_PACK_DIR = "/home/agent/tradingbot/reports/headroom_printer_packs/clio-ref-memory-activation-safe-20260709"
DEFAULT_PACK_HASH = "9cc7d677f43cd257bd28c0dbd3034b61559878103d13e5a2cdf8b9560f19d16b"

_BLOCKED_COMPONENTS = {
    ".env",
    "data",
    "logs",
    "secrets",
    "private",
    "credentials",
    "keys",
    "tokens",
}
_BLOCKED_SUFFIXES = {
    ".pem",
    ".key",
    ".crt",
    ".p12",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".conf",
    ".sql",
    ".xml",
    ".csv",
    ".sh",
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def headroom_context_enabled() -> bool:
    return _env_bool("CLIO_HEADROOM_CONTEXT_ENABLED", False)


def headroom_context_log_only() -> bool:
    return _env_bool("CLIO_HEADROOM_CONTEXT_LOG_ONLY", True)


def headroom_file_reads_enabled() -> bool:
    return _env_bool("CLIO_HEADROOM_FILE_READS_ENABLED", False)


def headroom_file_reads_log_only() -> bool:
    return _env_bool("CLIO_HEADROOM_FILE_READS_LOG_ONLY", True)


def _pack_dir() -> Path:
    return Path(os.getenv("CLIO_HEADROOM_PACK_DIR", DEFAULT_PACK_DIR)).resolve()


def _expected_pack_hash() -> str:
    return os.getenv("CLIO_HEADROOM_PACK_HASH", DEFAULT_PACK_HASH).strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blocked_source_path(path: str) -> Optional[str]:
    p = Path(path)
    parts = {part.lower() for part in p.parts}
    hit = parts & _BLOCKED_COMPONENTS
    if hit:
        return f"blocked path component: {sorted(hit)[0]}"
    if p.suffix.lower() in _BLOCKED_SUFFIXES:
        return f"blocked suffix: {p.suffix}"
    name = p.name.lower()
    if name.endswith(".env") or name.startswith(".env."):
        return "blocked env file"
    return None


def verify_headroom_pack(pack_dir: str | Path | None = None,
                         expected_pack_hash: str | None = None) -> dict[str, Any]:
    """Verify pack manifest, source/page hashes, and zero dropped chars."""
    root = Path(pack_dir).resolve() if pack_dir else _pack_dir()
    expected = _expected_pack_hash() if expected_pack_hash is None else (expected_pack_hash or "")
    manifest_path = root / "manifest.json"
    kept_path = root / "kept-text.ndjson"
    failures: list[str] = []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "available": False, "reason": f"manifest unreadable: {type(exc).__name__}", "failures": [str(exc)], "pack_dir": str(root)}

    if expected and manifest.get("packHash") != expected:
        failures.append(f"pack hash mismatch: {manifest.get('packHash')} != {expected}")

    totals = manifest.get("totals") or {}
    if int(totals.get("droppedChars") or 0) != 0:
        failures.append(f"dropped chars nonzero: {totals.get('droppedChars')}")
    if int(totals.get("pageCount") or 0) < 1:
        failures.append("no rendered pages")

    try:
        kept_text = kept_path.read_text(encoding="utf-8")
        kept_hash = _sha256_text(kept_text)
        if kept_hash != manifest.get("keptTextSha256"):
            failures.append(f"kept-text hash mismatch: {kept_hash} != {manifest.get('keptTextSha256')}")
    except Exception as exc:
        failures.append(f"kept-text unreadable: {type(exc).__name__}: {exc}")

    for source in manifest.get("sources", []):
        src = str(source.get("source") or "")
        blocked = _blocked_source_path(src)
        if blocked:
            failures.append(f"{blocked} for {src}")
            continue
        src_path = Path(src)
        try:
            actual = _sha256_file(src_path)
            if actual != source.get("sourceSha256"):
                failures.append(f"source hash mismatch for {src}")
        except Exception as exc:
            failures.append(f"source unreadable for {src}: {type(exc).__name__}: {exc}")

        for page in source.get("pages", []):
            page_file = str(page.get("file") or "")
            page_path = root / page_file
            try:
                actual_page = _sha256_file(page_path)
                if actual_page != page.get("sha256"):
                    failures.append(f"page hash mismatch for {page_file}")
                if int(page.get("width") or 0) <= 0 or int(page.get("height") or 0) <= 0 or int(page.get("bytes") or 0) <= 0:
                    failures.append(f"invalid page geometry for {page_file}")
            except Exception as exc:
                failures.append(f"page unreadable for {page_file}: {type(exc).__name__}: {exc}")

    return {
        "ok": not failures,
        "available": not failures,
        "reason": "ok" if not failures else "; ".join(failures[:3]),
        "failures": failures,
        "pack_dir": str(root),
        "manifest": manifest,
        "kept_text_path": str(kept_path),
    }


def _match_source_entry(manifest: dict[str, Any], target: str | Path) -> Optional[dict[str, Any]]:
    try:
        wanted = Path(target).resolve()
    except Exception:
        return None
    for source in manifest.get("sources", []):
        try:
            if Path(str(source.get("source") or "")).resolve() == wanted:
                return source
        except Exception:
            continue
    return None


def format_headroom_pack_reference(pack_dir: str | Path | None = None,
                                   system_label: str = "CLIO",
                                   max_sources: int | None = None) -> str:
    result = verify_headroom_pack(pack_dir)
    if not result.get("ok"):
        return ""

    manifest = result["manifest"]
    totals = manifest.get("totals") or {}
    estimates = manifest.get("estimates") or {}
    max_sources = max_sources if max_sources is not None else int(os.getenv("CLIO_HEADROOM_CONTEXT_MAX_SOURCES", "10"))
    root = Path(result["pack_dir"])

    lines = [
        f"--- HEADROOM IMAGE CONTEXT PACK ({system_label}) ---",
        "Use this pack for bulky ref/memory docs when visual context is acceptable.",
        "Original text files remain canonical. Images are lossy gist/prose context, not exact-string authority.",
        f"packName: {manifest.get('packName', 'unknown')}",
        f"packHash: {manifest.get('packHash', 'unknown')}",
        f"packDir: {root}",
        f"keptTextSidecar: {result['kept_text_path']} sha256={manifest.get('keptTextSha256', 'unknown')}",
        f"sources: {len(manifest.get('sources', []))} | pages: {totals.get('pageCount', 0)} | droppedChars: {totals.get('droppedChars', '?')}",
        f"estimated OpenAI prose-token reduction: {float(estimates.get('reductionOpenAIProse') or 0.0) * 100:.1f}%",
        "Representative page roots:",
    ]
    for source in manifest.get("sources", [])[:max_sources]:
        pages = source.get("pages", [])
        page_label = pages[0].get("file") if pages else "no-pages"
        lines.append(f"- {source.get('source')} -> {len(pages)} page(s), first={root / page_label}")
    if len(manifest.get("sources", [])) > max_sources:
        lines.append(f"- ... {len(manifest.get('sources', [])) - max_sources} more sources in manifest.json")
    lines.append("--- END HEADROOM IMAGE CONTEXT PACK ---")
    return "\n".join(lines)


def maybe_headroom_file_redirect(target: str | Path,
                                 display_path: str,
                                 exact_requested: bool = False,
                                 pack_dir: str | Path | None = None,
                                 system_label: str = "CLIO file tool") -> Optional[str]:
    """Return a compact image-pack pointer for whole-file reads when enabled."""
    if exact_requested or not headroom_file_reads_enabled() or headroom_file_reads_log_only():
        return None

    result = verify_headroom_pack(pack_dir)
    if not result.get("ok"):
        return None

    manifest = result["manifest"]
    source = _match_source_entry(manifest, target)
    if not source:
        return None

    root = Path(result["pack_dir"])
    lines = [
        f"HEADROOM PACK HIT ({system_label})",
        f"Requested file: {display_path}",
        f"Source sha256: {source.get('sourceSha256')}",
        f"Pack hash: {manifest.get('packHash')}",
        f"PROVENANCE: lossy image gist; canonical text remains {source.get('source')}.",
        "Raw text was not expanded. Use start_line/end_line for canonical text.",
        "Line reads bypass headroom redirects.",
        f"Exact-risk sidecar: {result['kept_text_path']} sha256={manifest.get('keptTextSha256')}",
        "Image pages:",
    ]
    for page in source.get("pages", []):
        page_path = root / str(page.get("file") or "")
        lines.append(
            f"- {page_path} sha256={page.get('sha256')} "
            f"{page.get('width')}x{page.get('height')}"
        )
    return "\n".join(lines)
