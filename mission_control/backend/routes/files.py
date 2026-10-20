"""File system API — read, write, list, search files across project directories."""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from mission_control.backend.auth import require_auth
from mission_control.backend.config import mc_config

logger = logging.getLogger("mission_control.files")
router = APIRouter(prefix="/api/files", tags=["files"])

# Allowed root directories — requests are sandboxed to these. Hydrinia's roots
# (tradingbot/.hydrinia) are permanently retired (operator direction, 2026-09-24)
# and go stale on disk: they are readable ONLY when the bridge is explicitly
# re-enabled, and are NEVER write/edit targets regardless of the bridge state.
_ALLOWED_ROOTS = (
    Path("/home/agent/openclaw"),
    Path("/home/agent/.claude"),
    Path("/etc/caddy"),
    Path("/etc/systemd/system"),
)
_HYDRINIA_READ_ONLY_ROOTS = (
    Path("/home/agent/tradingbot"),
    Path("/home/agent/.hydrinia"),
)


def _hydrinia_bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled

        return bridge_enabled()
    except Exception as e:
        logger.debug(f"files: bridge_enabled() unavailable ({type(e).__name__}: {e})")
        return False

# Max file size for read/write (10 MB)
_MAX_FILE_SIZE = 10 * 1024 * 1024

# Files/patterns that are NEVER readable (contain secrets)
_BLOCKED_FILENAMES = frozenset({
    ".env", ".env.bak", ".env.template", ".env.production", ".env.local",
    "cert.pem", "key.pem", "server.key", "server.crt",
    "credentials.json", "service_account.json",
})
_BLOCKED_SUFFIXES = (".pem", ".key", ".crt", ".p12", ".pfx", ".jks")

# Binary extensions to skip
_BINARY_EXTS = frozenset({
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll",
    ".whl", ".egg", ".tar", ".gz", ".bz2", ".zip", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svg",
    ".ttf", ".woff", ".woff2", ".eot",
    ".db", ".sqlite", ".sqlite3",
    ".pdf", ".doc", ".docx",
    ".mp3", ".mp4", ".wav", ".ogg",
})


def _is_blocked_file(p: Path) -> bool:
    """Check if a file should be blocked from read/write (contains secrets)."""
    if p.name in _BLOCKED_FILENAMES:
        return True
    if p.suffix.lower() in _BLOCKED_SUFFIXES:
        return True
    return False


def _resolve_safe(path_str: str, *, for_write: bool = False) -> Path:
    """Resolve a path and verify it falls within an allowed root.

    Hydrinia's roots are read-only, and only when the bridge is enabled —
    `for_write=True` (write/edit endpoints) never permits them.
    """
    p = Path(path_str).resolve()
    for root in _ALLOWED_ROOTS:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    if not for_write and _hydrinia_bridge_enabled():
        for root in _HYDRINIA_READ_ONLY_ROOTS:
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
    raise HTTPException(status_code=403, detail="Path outside allowed directories")


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTS


# ---- List directory ----

@router.get("/list")
async def list_directory(
    path: str = Query("/home/agent/openclaw", description="Directory to list"),
    _=Depends(require_auth),
):
    """List files and subdirectories in a directory."""
    dir_path = _resolve_safe(path)
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Not a directory")

    entries = []
    try:
        for entry in sorted(dir_path.iterdir()):
            name = entry.name
            # Skip common noise
            if name in ("__pycache__", ".git", "node_modules", ".venv", "venv"):
                continue
            info = {
                "name": name,
                "path": str(entry),
                "type": "dir" if entry.is_dir() else "file",
            }
            if entry.is_file():
                try:
                    info["size"] = entry.stat().st_size
                except OSError:
                    info["size"] = 0
            entries.append(info)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {"path": str(dir_path), "entries": entries}


# ---- Tree (recursive listing) ----

@router.get("/tree")
async def directory_tree(
    path: str = Query("/home/agent/openclaw", description="Root directory"),
    max_depth: int = Query(3, ge=1, le=6, description="Max recursion depth"),
    _=Depends(require_auth),
):
    """Recursive directory tree. Returns nested structure."""
    root = _resolve_safe(path)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Not a directory")

    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", ".next"}

    def _walk(dir_path: Path, depth: int) -> list:
        if depth > max_depth:
            return []
        items = []
        try:
            for entry in sorted(dir_path.iterdir()):
                if entry.name in skip_dirs:
                    continue
                if entry.name.startswith(".") and depth > 1:
                    continue
                node = {"name": entry.name, "path": str(entry)}
                if entry.is_dir():
                    node["type"] = "dir"
                    node["children"] = _walk(entry, depth + 1)
                else:
                    node["type"] = "file"
                    try:
                        node["size"] = entry.stat().st_size
                    except OSError:
                        node["size"] = 0
                items.append(node)
        except PermissionError:
            pass
        return items

    return {"root": str(root), "tree": _walk(root, 1)}


# ---- Read file ----

@router.get("/read")
def read_file(
    path: str = Query(..., description="Absolute file path"),
    offset: int = Query(0, ge=0, description="Line offset (0-based)"),
    limit: int = Query(0, ge=0, description="Max lines to return (0 = all)"),
    _=Depends(require_auth),
):
    """Read file contents. Returns text content with line numbers."""
    file_path = _resolve_safe(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if _is_blocked_file(file_path):
        raise HTTPException(status_code=403, detail="Access to this file is restricted")
    if _is_binary(file_path):
        return {"path": str(file_path), "binary": True, "size": file_path.stat().st_size}
    size = file_path.stat().st_size
    if size > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large ({size} bytes, max {_MAX_FILE_SIZE})")

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    lines = text.split("\n")
    total = len(lines)

    if offset > 0 or limit > 0:
        start = min(offset, total)
        end = start + limit if limit > 0 else total
        lines = lines[start:end]

    return {
        "path": str(file_path),
        "total_lines": total,
        "offset": offset,
        "lines_returned": len(lines),
        "content": "\n".join(lines),
    }


# ---- Write file ----

class WriteRequest(BaseModel):
    path: str
    content: str


@router.post("/write")
def write_file(
    req: WriteRequest,
    _=Depends(require_auth),
):
    """Write content to a file. Creates parent directories if needed. Uses atomic write."""
    file_path = _resolve_safe(req.path, for_write=True)
    if _is_blocked_file(file_path):
        raise HTTPException(status_code=403, detail="Cannot write to this file (restricted)")

    if len(req.content.encode("utf-8")) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Content too large")

    # Create parent dirs
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        tmp.write_text(req.content, encoding="utf-8")
        os.replace(str(tmp), str(file_path))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception:
        # Clean up tmp on failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to write file")

    logger.info(f"File written: {file_path} ({len(req.content)} chars)")
    return {"path": str(file_path), "size": len(req.content), "status": "ok"}


# ---- Edit file (find and replace) ----

class EditRequest(BaseModel):
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


@router.post("/edit")
def edit_file(
    req: EditRequest,
    _=Depends(require_auth),
):
    """Find-and-replace edit. Atomic write."""
    file_path = _resolve_safe(req.path, for_write=True)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if _is_blocked_file(file_path):
        raise HTTPException(status_code=403, detail="Cannot edit this file (restricted)")

    if not req.old_string:
        raise HTTPException(status_code=400, detail="old_string must not be empty")

    size = file_path.stat().st_size
    if size > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large for editing")

    try:
        content = file_path.read_text(encoding="utf-8")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    if req.old_string not in content:
        raise HTTPException(status_code=400, detail="old_string not found in file")

    if req.replace_all:
        new_content = content.replace(req.old_string, req.new_string)
        count = content.count(req.old_string)
    else:
        new_content = content.replace(req.old_string, req.new_string, 1)
        count = 1

    # Atomic write
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(str(tmp), str(file_path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to edit file")

    logger.info(f"File edited: {file_path} ({count} replacement(s))")
    return {"path": str(file_path), "replacements": count, "status": "ok"}


# ---- Search (grep-like) ----

@router.get("/search")
def search_files(
    query: str = Query(..., description="Search string or regex pattern"),
    path: str = Query("/home/agent/openclaw", description="Root directory to search"),
    glob: str = Query("*.py", description="File glob pattern"),
    max_results: int = Query(50, ge=1, le=200),
    _=Depends(require_auth),
):
    """Search file contents. Returns matching files and lines."""
    import re
    root = _resolve_safe(path)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Not a directory")

    if len(query) > 200:
        raise HTTPException(status_code=400, detail="Query too long (max 200 chars)")

    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)  # S153: escape to prevent ReDoS
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Invalid search query: {e}")

    results = []
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist"}

    for file_path in root.rglob(glob):
        if any(d in file_path.parts for d in skip_dirs):
            continue
        if _is_binary(file_path):
            continue
        if not file_path.is_file():
            continue
        if _is_blocked_file(file_path):
            continue
        try:
            if file_path.stat().st_size > _MAX_FILE_SIZE:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            continue
        matches = []
        for i, line in enumerate(text.split("\n"), 1):
            if pattern.search(line):
                matches.append({"line": i, "text": line.rstrip()[:200]})
                if len(matches) >= 5:  # Max 5 matches per file
                    break
        if matches:
            results.append({"path": str(file_path), "matches": matches})
            if len(results) >= max_results:
                break

    return {"query": query, "root": str(root), "glob": glob, "results": results}


# ---- File info / stat ----

@router.get("/info")
async def file_info(
    path: str = Query(..., description="File or directory path"),
    _=Depends(require_auth),
):
    """Get file/directory metadata."""
    p = _resolve_safe(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    stat = p.stat()
    return {
        "path": str(p),
        "type": "dir" if p.is_dir() else "file",
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_binary": _is_binary(p) if p.is_file() else False,
    }
