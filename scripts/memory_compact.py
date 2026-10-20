#!/usr/bin/env python3
"""Move-only MEMORY.md compactor for a Claude Code auto-memory directory (2026-09-28).

Claude Code loads only the head of MEMORY.md into each session (~25 KB / 200 lines). This tool
shrinks it by MOVING contiguous blocks of lines, byte-for-byte, into new `archive_*.md` files in
the SAME directory (so every relative link inside a moved block still resolves) and leaving a
one-line pointer where each block was. It never deletes, rewrites, reorders or summarizes a line
and never overwrites an existing file. survey and stage never write to the memory dir; apply
writes only new archive files and swaps MEMORY.md; rollback swaps the original back and renames
this run's files out. Every file it displaces is renamed into the run dir, never unlinked.

Subcommands (output: line numbers, headings, sizes, hashes and pointer lines; never body lines):
  survey    read-only: size, sha256, headings, heading-aligned windows with suggested byte targets
  stage     validate plan(s), build the result in a stage dir OUTSIDE the memory dir, prove it
            (byte-exact reconstruction, budget, floor, links), write manifest.json + report.md
  apply     re-verify the stage against the live directory, back up the whole memory dir
            (tar.gz on the same filesystem, verified member by member), create the archives
            (exclusive create), swap MEMORY.md in only if it still matches, re-verify
  rollback  idempotent and resumable: swap the original MEMORY.md back (only if it still
            matches what apply wrote) and rename this run's files into <run dir>/rolled_back

The MEMORY.md swap uses Linux renameat2(RENAME_EXCHANGE) when available: the old file survives
the swap and is compared after it, so an edit that lands in the last instant is swapped back,
not lost. Without it, the old inode is hard-linked into the run dir before os.replace, so a
writer that still has it open keeps writing into a file that survives; only an editor that
REPLACES the file (new inode) in a microsecond window could still lose an edit. apply and
rollback hold a non-blocking lock on the memory dir, so two runs never interleave. Close other
Claude Code sessions on the repo while applying.

Plan JSON (one per planner; several plans merge if their moves don't overlap):
  {"source_sha256": "<sha256 of MEMORY.md the plan was made against>",
   "window": {"start": 2, "end": 240},                       # optional; moves must stay inside
   "archives": {"archive_topic.md": {"description": "<one line>"}},
   "moves": [{"start": 120, "end": 245, "archive": "archive_topic.md",
              "pointer": "- [Title](archive_topic.md) — what is in it, when to read it"}]}
Lines are 1-based and inclusive. Line 1 (the index header) never moves.

Exit codes: 0 ok; 2 refused (validation/safety; nothing changed in the memory dir);
3 staged but not OK (over budget / under floor / no change / new broken links; apply refuses);
4 done (apply or rollback swapped MEMORY.md), but a later step reported a problem: read it;
5 STOP: a concurrent edit collided with the swap and could not be swapped back (nothing lost;
  the message says where each version is).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

INDEX_NAME = "MEMORY.md"
ARCHIVE_PAT = r"archive_[a-z0-9][a-z0-9_-]{0,80}\.md"
ARCHIVE_RE = re.compile(ARCHIVE_PAT)
POINTER_RE = re.compile(
    r"- \[(?P<title>[^\[\]<>`]{1,100})\]\((?P<archive>" + ARCHIVE_PAT + r")\) (?:—|--|-) (?P<hook>[^<>`]{1,200})")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
FENCE_RE = re.compile(rb"^[ ]{0,3}(```|~~~)")
LONE_CR_RE = re.compile(rb"\r(?!\n)")
RULE_WORDS_RE = re.compile(
    r"\b(never|always|must|do not|don't|rule|directive|standing|critical|secret|password|key)\b", re.I)
UNSAFE_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn", "Zl", "Zp"}
MARKER_PREFIX = "<!-- memory_compact:block "
TMP_PREFIX = f".{INDEX_NAME}.memory_compact."
DEFAULT_MAX_BYTES = 20000
DEFAULT_MAX_LINES = 180
DEFAULT_MIN_LINES = 20
MAX_POINTER_CHARS = 300
MAX_DESCRIPTION_CHARS = 200
_EXCHANGE_DISABLED = False  # tests flip this to exercise the os.replace fallback
ALLOW_FALLBACK_DEFAULT = False  # apply refuses without RENAME_EXCHANGE unless --allow-fallback-swap


class Refused(Exception):
    """A validation or safety check failed; the memory dir is as it was."""


class AppliedWithProblems(Exception):
    """MEMORY.md was swapped (apply or rollback happened), but a later step reported a problem."""


class SwapStuck(Exception):
    """A concurrent edit was detected and swapping it back failed; nothing is lost, but the
    message says where each version is and the operator must reconcile by hand."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def split_lines(data: bytes) -> list[bytes]:
    """Lines with their endings kept (b"\\n", b"\\r\\n" or b"\\r"); the last may have none."""
    return data.splitlines(keepends=True)


def resolved(path: Path) -> Path:
    return Path(os.path.realpath(path))


def is_inside(child: Path, parent: Path) -> bool:
    child, parent = resolved(child), resolved(parent)
    return child == parent or parent in child.parents


def memory_files(memory_dir: Path) -> dict[str, str]:
    """relative path -> sha256 for every regular file under the memory dir (symlinks not followed)."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(memory_dir, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name
            if p.is_symlink() or not p.is_file():
                continue
            out[str(p.relative_to(memory_dir))] = sha256_file(p)
    return out


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:  # e.g. ENAMETOOLONG on a hostile link target
        return False


def relative_links(text: str) -> list[str]:
    links = []
    for target in LINK_RE.findall(text):
        target = target.split("#", 1)[0]
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("/"):
            continue
        links.append(target)
    return links


def broken_links(text: str, base_dirs: list[Path]) -> list[str]:
    return sorted({t for t in relative_links(text) if not any(_exists(d / t) for d in base_dirs)})


def write_exclusive(path: Path, data: bytes, like: os.stat_result | None = None, on_created=None) -> None:
    """Create `path` (never overwrite), write, fsync. `like` copies mode (and owner when root).
    `on_created` runs the moment the file exists, so a failure mid-write is still undone."""
    mode = stat.S_IMODE(like.st_mode) if like is not None else 0o644
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        if on_created is not None:
            on_created()
        if like is not None:
            os.fchmod(fd, mode)
            if os.geteuid() == 0:
                try:
                    os.fchown(fd, like.st_uid, like.st_gid)
                except OSError:
                    pass
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, obj) -> None:
    """For the tool's own files in the run/stage dirs (never inside the memory dir)."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _fd_read_all(fd: int) -> bytes:
    chunks, off = [], 0
    while True:
        chunk = os.pread(fd, 1 << 20, off)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        off += len(chunk)


_RENAMEAT2 = None


def _load_renameat2():
    """Resolve renameat2 once (the process's own libc; no ldconfig subprocess). False when unavailable."""
    global _RENAMEAT2
    if _RENAMEAT2 is None:
        import ctypes
        try:
            fn = ctypes.CDLL(None, use_errno=True).renameat2
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            fn.restype = ctypes.c_int
            _RENAMEAT2 = fn
        except (OSError, AttributeError):
            _RENAMEAT2 = False
    return _RENAMEAT2


def _rename_exchange(a: Path, b: Path) -> bool:
    """Atomically swap two paths (Linux renameat2 RENAME_EXCHANGE). False when unsupported."""
    if _EXCHANGE_DISABLED:
        return False
    import ctypes
    fn = _load_renameat2()
    if not fn:
        return False
    at_fdcwd, rename_exchange = -100, 2
    if fn(at_fdcwd, os.fsencode(str(a)), at_fdcwd, os.fsencode(str(b)), rename_exchange) == 0:
        return True
    err = ctypes.get_errno()
    if err in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP):
        return False
    raise OSError(err, os.strerror(err), str(a))


def swap_index(memory_dir: Path, run_dir: Path, new_bytes: bytes, expected_sha: str, tag: str,
               created: list[str]) -> tuple[Path, list[str]]:
    """Put `new_bytes` at MEMORY.md only if MEMORY.md still hashes to `expected_sha`. The file it
    displaces is renamed to <run_dir>/MEMORY.md.displaced.<tag>, never unlinked. Returns
    (displaced path, warnings). Raises Refused only while nothing is swapped (the caller undoes
    `created`), or SwapStuck. Once the swap has happened it never raises: problems become warnings."""
    index = memory_dir / INDEX_NAME
    like = os.stat(index)
    tmp = memory_dir / f"{TMP_PREFIX}{tag}.tmp"
    displaced = run_dir / f"MEMORY.md.displaced.{tag}"
    _load_renameat2()   # resolve before the critical section
    write_exclusive(tmp, new_bytes, like, on_created=lambda: created.append(tmp.name))
    if sha256_file(index) != expected_sha:
        raise Refused("MEMORY.md changed just before the swap")
    if _rename_exchange(tmp, index):
        # MEMORY.md now holds new_bytes; tmp holds what MEMORY.md held at the instant of the swap.
        try:
            got = sha256_file(tmp)
        except OSError:
            got = None
        if got != expected_sha:
            try:
                swapped_back = _rename_exchange(tmp, index)
            except OSError as e:
                swapped_back, why = False, str(e)
            else:
                why = "RENAME_EXCHANGE unavailable"
            if not swapped_back:
                raise SwapStuck(f"a concurrent edit landed during the swap and swapping back failed ({why}): "
                                f"MEMORY.md holds the new index; the displaced file is at {tmp}")
            note = ""
            try:
                if sha256_file(tmp) != sha256_bytes(new_bytes):
                    note = (f"; ANOTHER edit landed on the staged copy during the swap-back and is preserved at "
                            f"{run_dir / ('undone_' + tmp.name)}: merge it by hand")
            except OSError:
                pass
            raise Refused("a concurrent edit landed during the swap (or the displaced file was unreadable); "
                          "swapped back, nothing lost" + note)
        created.remove(tmp.name)
        try:
            os.rename(tmp, displaced)
        except OSError as e:
            return tmp, [f"could not move the displaced original out of the memory dir ({e}); it is at {tmp}"]
        return displaced, []
    # No RENAME_EXCHANGE: hold the old inode open AND hard-link it into the run dir before the replace, so that any writer
    # that still has the old inode open (or opened it just before the swap) keeps writing into a file that survives.
    old_fd = os.open(index, os.O_RDONLY)
    try:
        if sha256_bytes(_fd_read_all(old_fd)) != expected_sha:
            raise Refused("MEMORY.md changed just before the swap")
        os.link(index, displaced)
        if os.fstat(old_fd).st_ino != os.stat(displaced).st_ino:      # an editor replaced the file between the read and the link
            os.rename(displaced, run_dir / f"undone_MEMORY.md.displaced.{tag}")
            raise Refused("MEMORY.md was replaced just before the swap; nothing changed")
        os.replace(tmp, index)
        created.remove(tmp.name)
        warnings: list[str] = []
        try:
            if sha256_file(displaced) != expected_sha:
                warnings.append(f"an edit landed in MEMORY.md during the swap; that version is saved at {displaced}")
        except OSError as e:
            warnings.append(f"could not re-read the displaced MEMORY.md ({e}); it is at {displaced}")
        return displaced, warnings
    finally:
        os.close(old_fd)


def undo_created(memory_dir: Path, run_dir: Path, created: list[str]) -> list[str]:
    """Rename every file this run created out of the memory dir (never delete). Returns errors."""
    errors = []
    for name in reversed(created):
        p = memory_dir / name
        if not os.path.lexists(p):
            continue
        try:
            os.rename(p, run_dir / f"undone_{name}")
        except OSError as e:
            errors.append(f"{name}: {e}")
    return errors


def load_memory_index(memory_dir: Path) -> bytes:
    index = memory_dir / INDEX_NAME
    if index.is_symlink() or not index.is_file():
        raise Refused(f"{index} is not a regular file")
    return index.read_bytes()


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate JSON keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


# ---------------------------------------------------------------------------
# survey
# ---------------------------------------------------------------------------

def survey(memory_dir: Path, windows: int, max_bytes: int, max_lines: int) -> dict:
    data = load_memory_index(memory_dir)
    lines = split_lines(data)
    headings = []
    for i, line in enumerate(lines, 1):
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        m = HEADING_RE.match(text)
        if m:
            headings.append({"line": i, "level": len(m.group(1)), "text": text[:120]})
    for k, h in enumerate(headings):
        end = headings[k + 1]["line"] - 1 if k + 1 < len(headings) else len(lines)
        h["end"] = end
        h["bytes"] = sum(len(lines[j]) for j in range(h["line"] - 1, end))
    windows = max(1, min(windows, 50))
    total = len(data)
    cut_candidates = [h["line"] for h in headings if h["line"] > 1]
    line_cum, cum = [], 0
    for ln in lines:
        cum += len(ln)
        line_cum.append(cum)
    wins, start = [], 1
    for w in range(1, windows):
        goal = total / windows * w
        best = None
        for c in cut_candidates:
            if c <= start:
                continue
            if best is None or abs(line_cum[c - 2] - goal) < abs(line_cum[best - 2] - goal):
                best = c
        if best is None:
            break
        wins.append((start, best - 1))
        start = best
    if lines:
        wins.append((start, len(lines)))
    keep_budget = max_bytes * 0.9
    win_out = []
    for n, (s, e) in enumerate(wins, 1):
        b = sum(len(lines[j]) for j in range(s - 1, e))
        win_out.append({"window": n, "start": s, "end": e, "lines": e - s + 1, "bytes": b,
                        "suggested_keep_bytes": int(keep_budget * b / total) if total else 0})
    return {
        "memory_dir": str(memory_dir), "index": INDEX_NAME, "sha256": sha256_bytes(data),
        "bytes": total, "lines": len(lines), "ends_with_newline": data.endswith(b"\n"),
        "lone_cr": len(LONE_CR_RE.findall(data)),
        "marker_like_lines": sum(1 for ln in lines if ln.startswith(MARKER_PREFIX.encode())),
        "budget": {"max_bytes": max_bytes, "max_lines": max_lines,
                   "over_by_bytes": max(0, total - max_bytes), "over_by_lines": max(0, len(lines) - max_lines)},
        "existing_files": sorted(memory_files(memory_dir)),
        "existing_archives": sorted(p.name for p in memory_dir.glob("archive_*.md")),
        "headings": headings, "windows": win_out,
    }


# ---------------------------------------------------------------------------
# plan validation
# ---------------------------------------------------------------------------

def _safe_text(value, what: str, max_chars: int, forbid: str = "") -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"{what} must be a non-empty string")
    if len(value) > max_chars:
        raise Refused(f"{what} must be at most {max_chars} characters")
    bad = [c for c in value if unicodedata.category(c) in UNSAFE_CATEGORIES or c in forbid]
    if bad:
        raise Refused(f"{what} contains control/format/private-use characters or one of {forbid!r}: "
                      f"{sorted({hex(ord(c)) for c in bad})[:5]}")
    return value


def load_plans(plan_paths: list[Path], source_sha: str, n_lines: int, memory_dir: Path) -> tuple[dict, list[dict]]:
    archives: dict[str, dict] = {}
    moves: list[dict] = []
    for pp in plan_paths:
        try:
            plan = json.loads(pp.read_text(), object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError, RecursionError) as e:
            raise Refused(f"{pp}: cannot read plan JSON: {str(e)[:200]}")
        if not isinstance(plan, dict):
            raise Refused(f"{pp}: plan must be a JSON object")
        unknown = set(plan) - {"source_sha256", "window", "archives", "moves", "notes"}
        if unknown:
            raise Refused(f"{pp}: unknown plan keys {sorted(map(str, unknown))}")
        if plan.get("source_sha256") != source_sha:
            raise Refused(f"{pp}: source_sha256 does not match the current MEMORY.md "
                          f"(plan made against another version; re-survey and re-plan)")
        lo, hi = 2, n_lines
        if plan.get("window") is not None:
            w = plan["window"]
            if not (isinstance(w, dict) and set(w) == {"start", "end"}
                    and all(type(w[k]) is int for k in ("start", "end")) and 1 <= w["start"] <= w["end"] <= n_lines):
                raise Refused(f"{pp}: window must be {{start, end}} within 1..{n_lines}")
            lo, hi = max(2, w["start"]), w["end"]
        plan_archives = plan.get("archives") or {}
        if not isinstance(plan_archives, dict):
            raise Refused(f"{pp}: archives must be an object")
        for name, meta in plan_archives.items():
            if not isinstance(name, str) or not ARCHIVE_RE.fullmatch(name):
                raise Refused(f"{pp}: archive name {name!r} must match {ARCHIVE_PAT}")
            if os.path.lexists(memory_dir / name):
                raise Refused(f"{pp}: {name} already exists in the memory dir (never overwritten); pick a new name")
            if not isinstance(meta, dict) or set(meta) != {"description"}:
                raise Refused(f"{pp}: archives[{name!r}] must be {{\"description\": ...}}")
            desc = _safe_text(meta["description"], f"archives[{name!r}].description", MAX_DESCRIPTION_CHARS, "<>`")
            if name in archives and archives[name]["description"] != desc:
                raise Refused(f"{pp}: archive {name} is declared by two plans with different descriptions")
            archives[name] = {"description": desc}
        plan_moves = plan.get("moves")
        if not isinstance(plan_moves, list) or not plan_moves:
            raise Refused(f"{pp}: moves must be a non-empty list")
        for k, mv in enumerate(plan_moves):
            where = f"{pp}: moves[{k}]"
            if not isinstance(mv, dict) or not {"start", "end", "archive", "pointer"} <= set(mv) \
                    or set(mv) - {"start", "end", "archive", "pointer", "why"}:
                raise Refused(f"{where}: must be {{start, end, archive, pointer[, why]}}")
            s, e = mv["start"], mv["end"]
            if type(s) is not int or type(e) is not int:
                raise Refused(f"{where}: start/end must be integers")
            if s == 1:
                raise Refused(f"{where}: line 1 (the index header) never moves")
            if not (lo <= s <= e <= hi):
                raise Refused(f"{where}: lines {s}-{e} outside {lo}-{hi}")
            name = mv["archive"]
            if not isinstance(name, str) or name not in plan_archives:
                raise Refused(f"{where}: archive {name!r} is not declared in this plan's archives")
            pointer = _safe_text(mv["pointer"], f"{where}.pointer", MAX_POINTER_CHARS)
            m = POINTER_RE.fullmatch(pointer)
            if not m or m.group("archive") != name:
                raise Refused(f"{where}: pointer must be exactly '- [Title](<its archive>) — hook' "
                              f"(no < > ` [ ] in the title, no < > ` in the hook)")
            why = mv.get("why", "")
            why = _safe_text(why, f"{where}.why", 300) if why != "" else ""
            moves.append({"start": s, "end": e, "archive": name, "pointer": pointer, "why": why, "plan": str(pp)})
    moves.sort(key=lambda m: m["start"])
    for a, b in zip(moves, moves[1:]):
        if b["start"] <= a["end"]:
            raise Refused(f"moves overlap: lines {a['start']}-{a['end']} ({a['plan']}) and "
                          f"{b['start']}-{b['end']} ({b['plan']})")
    unused = sorted(set(archives) - {m["archive"] for m in moves})
    if unused:
        raise Refused(f"archives declared but never used: {unused}")
    return archives, moves


# ---------------------------------------------------------------------------
# build + verify (pure functions over bytes)
# ---------------------------------------------------------------------------

def _marker(block_no: int, mv: dict, block_sha: str) -> bytes:
    return f"{MARKER_PREFIX}{block_no} lines {mv['start']}-{mv['end']} sha256={block_sha} -->\n".encode()


def build(original: bytes, archives: dict, moves: list[dict], stamp: str) -> tuple[bytes, dict[str, bytes], list[dict]]:
    lines = split_lines(original)
    src_sha = sha256_bytes(original)
    index_out: list[bytes] = []
    blocks: list[dict] = []
    cursor = 1
    for n, mv in enumerate(moves):
        index_out.extend(lines[cursor - 1:mv["start"] - 1])
        block = b"".join(lines[mv["start"] - 1:mv["end"]])
        blocks.append({"block": n, "start": mv["start"], "end": mv["end"], "archive": mv["archive"],
                       "pointer": mv["pointer"], "why": mv["why"], "block_sha256": sha256_bytes(block),
                       "block_bytes": len(block), "pointer_index_line": len(index_out) + 1,
                       "eof_no_newline": not block.endswith((b"\n", b"\r"))})
        index_out.append(mv["pointer"].encode("utf-8") + b"\n")
        cursor = mv["end"] + 1
    index_out.extend(lines[cursor - 1:])
    archive_bytes: dict[str, bytes] = {}
    for name, meta in archives.items():
        mine = [b for b in blocks if b["archive"] == name]
        ranges = ", ".join(f"{b['start']}-{b['end']}" for b in mine)
        header = (f"---\nname: {name[:-3]}\ndescription: {json.dumps(meta['description'], ensure_ascii=False)}\n"
                  f"type: reference\n---\n"
                  f"<!-- Moved verbatim from {INDEX_NAME} by scripts/memory_compact.py on {stamp}; "
                  f"source sha256 {src_sha}; original lines {ranges}. Block markers below are "
                  f"used by rollback checks: leave them in place. -->\n\n").encode("utf-8")
        body = [header]
        for b in mine:
            block = b"".join(lines[b["start"] - 1:b["end"]])
            body.append(_marker(b["block"], b, b["block_sha256"]))
            body.append(block + b"\n" if b["eof_no_newline"] else block)
        archive_bytes[name] = b"".join(body)
    return b"".join(index_out), archive_bytes, blocks


def extract_blocks(archive: bytes) -> dict[int, bytes]:
    """block number -> bytes between its marker and the next marker (or EOF)."""
    out: dict[int, bytes] = {}
    current, buf = None, []
    for line in split_lines(archive):
        if line.startswith(MARKER_PREFIX.encode()):
            if current is not None:
                out[current] = b"".join(buf)
            m = re.match(rb"<!-- memory_compact:block (\d+) ", line)
            if not m:
                raise Refused("malformed block marker")
            current, buf = int(m.group(1)), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = b"".join(buf)
    return out


def verify(original: bytes, index: bytes, archive_bytes: dict[str, bytes], blocks: list[dict]) -> None:
    """Reconstruct the original byte-for-byte from the index + archives, or raise Refused."""
    idx_lines = split_lines(index)
    extracted: dict[int, bytes] = {}
    for data in archive_bytes.values():
        for n, blk in extract_blocks(data).items():
            if n in extracted:
                raise Refused(f"block {n} appears twice")
            extracted[n] = blk
    if sorted(extracted) != [b["block"] for b in blocks]:
        raise Refused("archive blocks do not match the plan")
    by_line = {}
    for b in blocks:
        blk = extracted[b["block"]]
        if b["eof_no_newline"]:
            if not blk.endswith(b"\n"):
                raise Refused(f"block {b['block']}: missing the separator newline")
            blk = blk[:-1]
        if sha256_bytes(blk) != b["block_sha256"]:
            raise Refused(f"block {b['block']} (lines {b['start']}-{b['end']}) is not byte-identical to the original")
        pl = b["pointer_index_line"]
        if pl > len(idx_lines) or idx_lines[pl - 1] != b["pointer"].encode("utf-8") + b"\n":
            raise Refused(f"pointer for block {b['block']} is not at index line {pl}")
        by_line[pl] = blk
    rebuilt = b"".join(by_line.get(i, ln) for i, ln in enumerate(idx_lines, 1))
    if rebuilt != original:
        raise Refused("reconstruction from the index + archives does not equal the original MEMORY.md")


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------

def stage(memory_dir: Path, plan_paths: list[Path], stage_dir: Path, max_bytes: int, max_lines: int,
          min_lines: int = DEFAULT_MIN_LINES) -> dict:
    if is_inside(stage_dir, memory_dir) or is_inside(memory_dir, stage_dir):
        raise Refused("the stage dir must be outside the memory dir (and not contain it)")
    if stage_dir.is_symlink() or (stage_dir.exists() and (not stage_dir.is_dir() or any(stage_dir.iterdir()))):
        raise Refused(f"stage dir {stage_dir} must be a new or empty directory (never overwritten)")
    before = memory_files(memory_dir)
    original = load_memory_index(memory_dir)
    src_sha = sha256_bytes(original)
    orig_lines = split_lines(original)
    if LONE_CR_RE.search(original):
        raise Refused("MEMORY.md contains a bare CR line ending: line numbers would not match editors or "
                      "the Read tool; fix the line endings by hand first")
    clash = [i for i, ln in enumerate(orig_lines, 1) if ln.startswith(MARKER_PREFIX.encode())]
    if clash:
        raise Refused(f"MEMORY.md lines {clash[:10]} already look like memory_compact block markers")
    archives, moves = load_plans(plan_paths, src_sha, len(orig_lines), memory_dir)
    inside_before, inside = [], False  # inside_before[i-1]: is line i inside a code fence?
    for ln in orig_lines:
        inside_before.append(inside)
        if FENCE_RE.match(ln):
            inside = not inside
    inside_before.append(inside)  # state after the last line
    for mv in moves:
        if inside_before[mv["start"] - 1] or inside_before[mv["end"]]:
            raise Refused(f"lines {mv['start']}-{mv['end']} start or end inside a code fence; "
                          f"move whole fenced blocks or none of them")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index, archive_bytes, blocks = build(original, archives, moves, stamp)
    verify(original, index, archive_bytes, blocks)  # in memory

    stage_dir.mkdir(parents=True, exist_ok=True)
    write_exclusive(stage_dir / INDEX_NAME, index)
    for name, data in archive_bytes.items():
        write_exclusive(stage_dir / name, data)
    index_disk = (stage_dir / INDEX_NAME).read_bytes()
    arch_disk = {name: (stage_dir / name).read_bytes() for name in archive_bytes}
    verify(original, index_disk, arch_disk, blocks)  # again, from disk

    sensitive = []
    for b in blocks:
        hits = [i for i in range(b["start"], b["end"] + 1)
                if RULE_WORDS_RE.search(orig_lines[i - 1].decode("utf-8", "replace"))]
        if hits:
            sensitive.append({"block": b["block"], "start": b["start"], "end": b["end"],
                              "rule_word_lines": hits[:40], "count": len(hits)})
    bases = [memory_dir, stage_dir]
    pre_broken = broken_links(original.decode("utf-8", "replace"), [memory_dir])
    post_broken = sorted(set(broken_links(index_disk.decode("utf-8", "replace"), bases)).union(
        *[broken_links(d.decode("utf-8", "replace"), bases) for d in arch_disk.values()]))
    new_broken = sorted(set(post_broken) - set(pre_broken))
    n_index_lines = len(split_lines(index_disk))
    within = len(index_disk) <= max_bytes and n_index_lines <= max_lines
    floor_ok = n_index_lines >= min(min_lines, len(orig_lines))
    if memory_files(memory_dir) != before:
        raise Refused("the memory dir changed while staging; re-run survey/plan/stage")
    problems = ([] if within else ["over budget"]) + ([] if floor_ok else [f"fewer than {min_lines} lines kept"]) \
        + ([f"new broken links {new_broken}"] if new_broken else []) \
        + (["no change: the new MEMORY.md equals the original"] if index_disk == original else [])
    manifest = {
        "tool": "scripts/memory_compact.py", "staged_at": stamp, "memory_dir": str(resolved(memory_dir)),
        "source_sha256": src_sha, "source_bytes": len(original), "source_lines": len(orig_lines),
        "index_sha256": sha256_bytes(index_disk), "index_bytes": len(index_disk), "index_lines": n_index_lines,
        "budget": {"max_bytes": max_bytes, "max_lines": max_lines, "min_lines": min_lines,
                   "within": within, "floor_ok": floor_ok},
        "archives": {n: {"sha256": sha256_bytes(d), "bytes": len(d), "description": archives[n]["description"]}
                     for n, d in arch_disk.items()},
        "blocks": blocks, "plans": [str(p) for p in plan_paths],
        "links": {"broken_before": pre_broken, "broken_after": post_broken, "new_broken": new_broken},
        "sensitive_moves": sensitive, "problems": problems, "ok": not problems,
    }
    write_json_atomic(stage_dir / "manifest.json", manifest)
    (stage_dir / "report.md").write_text(render_report(manifest))
    return manifest


def render_report(m: dict) -> str:
    out = [f"# memory_compact stage report ({m['staged_at']})", "",
           f"- MEMORY.md: {m['source_bytes']} B / {m['source_lines']} lines -> "
           f"{m['index_bytes']} B / {m['index_lines']} lines "
           f"(budget {m['budget']['max_bytes']} B / {m['budget']['max_lines']} lines, "
           f"floor {m['budget']['min_lines']} lines)",
           "- Reconstruction: byte-identical (verified in memory and from disk)",
           f"- New broken links: {m['links']['new_broken'] or 'none'}"
           f" (already broken before: {len(m['links']['broken_before'])})",
           f"- Result: {'OK to apply' if m['ok'] else 'NOT OK, apply will refuse: ' + '; '.join(m['problems'])}", "",
           "## Moves (original line ranges -> archive; the pointer line that replaces each block)"]
    for b in m["blocks"]:
        out.append(f"- block {b['block']}: lines {b['start']}-{b['end']} ({b['block_bytes']} B) -> {b['archive']}; "
                   f"new line {b['pointer_index_line']}; why: {b['why'] or '-'}")
        out.append(f"  pointer: {b['pointer']}")
    out += ["", "## Moves containing rule-like words (reviewers: confirm each is superseded or duplicated elsewhere)"]
    if not m["sensitive_moves"]:
        out.append("- none")
    for s in m["sensitive_moves"]:
        out.append(f"- block {s['block']} (lines {s['start']}-{s['end']}): {s['count']} line(s), e.g. {s['rule_word_lines'][:10]}")
    out += ["", "## Archives created"]
    for n, a in m["archives"].items():
        out.append(f"- {n}: {a['bytes']} B; {a['description']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# apply / rollback
# ---------------------------------------------------------------------------

def _leftover_tmps(memory_dir: Path) -> list[str]:
    return sorted(p.name for p in memory_dir.iterdir() if p.name.startswith(TMP_PREFIX))


def _new_run_dir(backup_root: Path, memory_dir: Path) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = backup_root / f"memory_compact_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if os.stat(run_dir).st_dev != os.stat(memory_dir).st_dev:
        raise Refused(f"the backup dir must be on the same filesystem as the memory dir (every move must be a "
                      f"rename, never copy+delete); {run_dir} is empty and can be removed")
    return run_id, run_dir


def _lock_dir(memory_dir: Path) -> int:
    """Advisory exclusive lock on the memory dir itself (no file created). Serialises apply/rollback runs of this tool."""
    fd = os.open(memory_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise Refused("another memory_compact apply/rollback is running on this memory dir; wait for it to finish")
    return fd


def apply(memory_dir: Path, stage_dir: Path, backup_root: Path, allow_fallback: bool | None = None) -> dict:
    fd = _lock_dir(memory_dir)
    try:
        return _apply(memory_dir, stage_dir, backup_root,
                      ALLOW_FALLBACK_DEFAULT if allow_fallback is None else allow_fallback)
    finally:
        os.close(fd)


def rollback(memory_dir: Path, run_dir: Path) -> dict:
    fd = _lock_dir(memory_dir)
    try:
        return _rollback(memory_dir, run_dir)
    finally:
        os.close(fd)


def _exchange_supported_in(run_dir: Path) -> bool:
    """Probe RENAME_EXCHANGE on the run dir (same filesystem as the memory dir). The two probe
    files stay in the run dir: this tool never deletes."""
    a, b = run_dir / "exchange_probe_a", run_dir / "exchange_probe_b"
    write_exclusive(a, b"a")
    write_exclusive(b, b"b")
    try:
        return _rename_exchange(a, b) and a.read_bytes() == b"b"
    except OSError:
        return False


def _apply(memory_dir: Path, stage_dir: Path, backup_root: Path, allow_fallback: bool = False) -> dict:
    if is_inside(backup_root, memory_dir) or is_inside(memory_dir, backup_root):
        raise Refused("the backup dir must be outside the memory dir (and not contain it)")
    try:
        m = json.loads((stage_dir / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {stage_dir / 'manifest.json'}: {e}")
    if not m.get("ok"):
        raise Refused(f"the stage is not OK ({'; '.join(m.get('problems') or ['unknown'])}); apply refuses")
    if resolved(memory_dir) != Path(m["memory_dir"]):
        raise Refused("the stage was made for a different memory dir")
    leftovers = _leftover_tmps(memory_dir)
    if leftovers:
        raise Refused(f"leftover temp file(s) from an interrupted run: {leftovers}; run that run's rollback "
                      f"(or move them out by hand) first")
    live = load_memory_index(memory_dir)
    if sha256_bytes(live) != m["source_sha256"]:
        raise Refused("MEMORY.md changed since staging; re-run survey, plan and stage")
    staged_index = (stage_dir / INDEX_NAME).read_bytes()
    if sha256_bytes(staged_index) != m["index_sha256"]:
        raise Refused("staged MEMORY.md does not match the manifest (edited after staging)")
    staged_arch = {}
    for name, meta in m["archives"].items():
        if not isinstance(name, str) or not ARCHIVE_RE.fullmatch(name):
            raise Refused(f"bad archive name in manifest: {name!r}")
        data = (stage_dir / name).read_bytes()
        if sha256_bytes(data) != meta["sha256"]:
            raise Refused(f"staged {name} does not match the manifest (edited after staging)")
        if os.path.lexists(memory_dir / name):
            raise Refused(f"{name} now exists in the memory dir; refusing to overwrite")
        staged_arch[name] = data
    verify(live, staged_index, staged_arch, m["blocks"])

    run_id, run_dir = _new_run_dir(backup_root, memory_dir)
    if not allow_fallback and not _exchange_supported_in(run_dir):
        raise Refused("this filesystem has no renameat2(RENAME_EXCHANGE), so the swap would use the os.replace "
                      "fallback, where an editor replacing MEMORY.md in the same microsecond could lose its edit. "
                      "Nothing changed. Re-run with --allow-fallback-swap only if that is acceptable.")
    before = memory_files(memory_dir)
    tar_path = run_dir / "memory_dir.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in before:
            tar.add(memory_dir / rel, arcname=rel, recursive=False)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {mm.name: mm for mm in tar.getmembers() if mm.isfile()}
        if set(members) != set(before):
            raise Refused("backup tar does not contain exactly the memory dir's files; nothing applied")
        for rel, digest in before.items():
            f = tar.extractfile(members[rel])
            if f is None or sha256_bytes(f.read()) != digest:
                raise Refused(f"backup tar copy of {rel} does not verify; nothing applied")
    if before.get(INDEX_NAME) != m["source_sha256"] or sha256_file(memory_dir / INDEX_NAME) != m["source_sha256"]:
        raise Refused("MEMORY.md changed during the backup; nothing applied")
    write_exclusive(run_dir / "MEMORY.md.orig", live)
    write_json_atomic(run_dir / "backup_manifest.json", {
        "files": before,
        "symlinks_not_backed_up": sorted(str(p.relative_to(memory_dir)) for p in memory_dir.rglob("*") if p.is_symlink()),
    })
    run = {"run_id": run_id, "state": "pending", "memory_dir": str(resolved(memory_dir)),
           "stage_dir": str(resolved(stage_dir)), "original_sha256": m["source_sha256"],
           "new_index_sha256": m["index_sha256"],
           "created": {n: meta["sha256"] for n, meta in m["archives"].items()},
           "tmp_prefix": f"{TMP_PREFIX}{run_id}", "backup_tar": str(tar_path)}
    write_json_atomic(run_dir / "run.json", run)

    like = os.stat(memory_dir / INDEX_NAME)
    created: list[str] = []
    try:
        for name, data in staged_arch.items():
            write_exclusive(memory_dir / name, data, like, on_created=lambda n=name: created.append(n))
        displaced, warnings = swap_index(memory_dir, run_dir, staged_index, m["source_sha256"], run_id, created)
    except SwapStuck:
        raise
    except BaseException as e:
        try:  # decide from the file, never from where the exception came from
            swapped = sha256_file(memory_dir / INDEX_NAME) == m["index_sha256"]
        except OSError:
            swapped = False
        if swapped:
            if not isinstance(e, Exception):
                raise
            raise AppliedWithProblems(f"MEMORY.md was swapped but apply was interrupted ({type(e).__name__}: {e}); "
                                      f"inspect, then rollback --run-dir {run_dir} if needed")
        errors = undo_created(memory_dir, run_dir, created)
        if not isinstance(e, Exception):
            raise
        tail = f"; undo problems: {errors}" if errors else ""
        raise Refused(f"apply aborted and undone ({type(e).__name__}: {e}); MEMORY.md was not replaced{tail}")

    problems = list(warnings)
    try:
        run.update(state="applied", displaced=str(displaced))
        write_json_atomic(run_dir / "run.json", run)
        try:
            if sha256_file(displaced) != m["source_sha256"]:
                problems.append(f"a write landed in the displaced original after the swap; it is preserved at "
                                f"{displaced}: merge it into MEMORY.md by hand")
        except OSError:
            pass
        after = memory_files(memory_dir)
        expected = dict(before)
        expected[INDEX_NAME] = m["index_sha256"]
        expected.update({n: meta["sha256"] for n, meta in m["archives"].items()})
        if after != expected:
            diff = sorted(k for k in set(after) | set(expected) if after.get(k) != expected.get(k))
            problems.append(f"the memory dir differs from what was staged at {diff[:10]}")
    except Exception as e:
        problems.append(f"post-apply step failed: {type(e).__name__}: {e}")
    result = {"run_dir": str(run_dir), **run}
    if problems:
        raise AppliedWithProblems(json.dumps({"problems": problems, **result}, indent=1))
    return result


def _rollback(memory_dir: Path, run_dir: Path) -> dict:
    """Idempotent and resumable. Decides from the files, not from run.json's state."""
    try:
        run = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {run_dir / 'run.json'}: {e}")
    try:
        created_ok = isinstance(run["created"], dict) and all(
            isinstance(n, str) and ARCHIVE_RE.fullmatch(n) and isinstance(d, str) for n, d in run["created"].items())
        prefix_ok = isinstance(run["tmp_prefix"], str) and run["tmp_prefix"] == f"{TMP_PREFIX}{run['run_id']}" \
            and re.fullmatch(r"\d{8}T\d{12}Z", str(run["run_id"]))
        if not (created_ok and prefix_ok):
            raise Refused(f"{run_dir / 'run.json'} fails validation; refusing to act on it")
        if resolved(memory_dir) != Path(run["memory_dir"]):
            raise Refused("this run was for a different memory dir")
    except (KeyError, TypeError) as e:
        raise Refused(f"{run_dir / 'run.json'} is missing or has malformed fields ({e})")
    orig = (run_dir / "MEMORY.md.orig").read_bytes()
    if sha256_bytes(orig) != run["original_sha256"]:
        raise Refused("MEMORY.md.orig in the run dir does not verify")
    cur = sha256_file(memory_dir / INDEX_NAME)
    if cur == run["new_index_sha256"]:
        need_restore = True
    elif cur == run["original_sha256"]:
        need_restore = False  # never swapped (interrupted apply) or already restored
    else:
        raise Refused("MEMORY.md was edited after apply; save the new edits, then restore by hand from "
                      f"{run_dir / 'MEMORY.md.orig'} (the tar in the run dir has every original file)")
    parked = run_dir / "rolled_back"
    to_move, left_alone = [], []
    for name, digest in run["created"].items():
        live = memory_dir / name
        if os.path.lexists(live):
            ours = not live.is_symlink() and live.is_file() and sha256_file(live) == digest
            if need_restore and not ours:
                raise Refused(f"{name} was edited after apply; resolve by hand (original is in MEMORY.md.orig)")
            if ours or run.get("state") == "pending":
                to_move.append(name)  # after an interrupted apply it may be partial: still this run's file
            else:
                left_alone.append(name)  # appeared after this run finished: not ours
    to_move += [p.name for p in memory_dir.iterdir() if p.name.startswith(run["tmp_prefix"])]
    parked.mkdir(exist_ok=True)
    warnings: list[str] = []
    if need_restore:
        created: list[str] = []
        tag = f"{run['run_id']}_rollback_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            _, warnings = swap_index(memory_dir, run_dir, orig, run["new_index_sha256"], tag, created)
        except (Refused, OSError) as e:
            errors = undo_created(memory_dir, run_dir, created)
            raise Refused(f"rollback aborted ({e}); MEMORY.md unchanged{'; undo problems: ' + str(errors) if errors else ''}")
    problems = list(warnings)
    moved = []
    for name in to_move:
        dest = parked / name
        if os.path.lexists(dest):
            dest = parked / f"{name}.{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        try:
            os.rename(memory_dir / name, dest)
            moved.append(name)
        except OSError as e:
            problems.append(f"could not move {name} out ({e}); re-run rollback")
    result = {"restored": need_restore, "moved_to": str(parked), "moved": sorted(moved),
              "left_alone": sorted(left_alone)}
    if not problems:
        try:
            run["state"] = "rolled_back"
            write_json_atomic(run_dir / "run.json", run)
        except OSError as e:
            problems.append(f"could not record the rollback in run.json ({e})")
    if problems:
        raise AppliedWithProblems(json.dumps({"problems": problems, **result}, indent=1))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--memory-dir", required=True, type=Path)
    s.add_argument("--windows", type=int, default=1)
    s.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    s.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    st = sub.add_parser("stage")
    st.add_argument("--memory-dir", required=True, type=Path)
    st.add_argument("--plan", required=True, type=Path, action="append")
    st.add_argument("--stage-dir", required=True, type=Path)
    st.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    st.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    st.add_argument("--min-lines", type=int, default=DEFAULT_MIN_LINES)
    a = sub.add_parser("apply")
    a.add_argument("--memory-dir", required=True, type=Path)
    a.add_argument("--stage-dir", required=True, type=Path)
    a.add_argument("--backup-dir", required=True, type=Path)
    a.add_argument("--allow-fallback-swap", action="store_true",
                   help="accept the os.replace fallback when the filesystem lacks RENAME_EXCHANGE")
    r = sub.add_parser("rollback")
    r.add_argument("--memory-dir", required=True, type=Path)
    r.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "survey":
            print(json.dumps(survey(args.memory_dir, args.windows, args.max_bytes, args.max_lines), indent=1))
            return 0
        if args.cmd == "stage":
            m = stage(args.memory_dir, args.plan, args.stage_dir, args.max_bytes, args.max_lines, args.min_lines)
            print(render_report(m))
            return 0 if m["ok"] else 3
        if args.cmd == "apply":
            res = apply(args.memory_dir, args.stage_dir, args.backup_dir, args.allow_fallback_swap or None)
            print(json.dumps(res, indent=1))
            print(f"\nRollback: python3 scripts/memory_compact.py rollback --memory-dir {args.memory_dir} "
                  f"--run-dir {res['run_dir']}")
            return 0
        print(json.dumps(rollback(args.memory_dir, args.run_dir), indent=1))
        return 0
    except AppliedWithProblems as e:
        print(f"DONE, BUT CHECK THESE PROBLEMS (the swap happened; rollback works):\n{e}", file=sys.stderr)
        return 4
    except SwapStuck as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 5
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # never a raw traceback; the memory dir is unchanged unless a message above says so
        print(f"REFUSED: unexpected {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
