# agent/kronos/vendor — provenance pin (CLIO Kronos, K1 build)

**Status (2026-09-26):** COPIED from Hydrinia's already-provisioned
`kronos_vendor/` tree (`/home/agent/tradingbot/kronos_vendor/` on the VPS;
this build read it read-only at
`/home/user/hydrinia/kronos_vendor/`). Hydrinia's own
`kronos_vendor/VENDOR_PIN.md` (2026-06-10, later amended) is the original
provisioning record — the pin, tarball hash, and both patches below are
copied from it verbatim, not re-derived. Hydrinia is retired
(`CLIO_HYDRINIA_BRIDGE_ENABLED=OFF`); this is CLIO's own copy so Kronos has
no runtime dependency on the Hydrinia checkout.

## 1. Pin (copied from Hydrinia's record)

- **Upstream:** `github.com/shiyu-coder/Kronos`
- **Pinned commit:** `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- **Tarball:** `https://github.com/shiyu-coder/Kronos/archive/67b630e67f6a18c9e9be918d9b4337c960db1e9a.tar.gz`
- **Tarball sha256:** `969719e47b2134d8a56533784508b6d859bd1c9aacb1b62e4a504cb4fc096021`
- **License:** MIT (`Copyright (c) 2025 ShiYu`) — retained unchanged at
  `agent/kronos/vendor/LICENSE`.

## 2. What is vendored (and what is excluded)

Copied from Hydrinia's tree:

- `model/` — `__init__.py`, `kronos.py`, `module.py` (the model, tokenizer,
  predictor), byte-identical to Hydrinia's already-patched copies (verified:
  both required patches, below, were already applied there).
- `LICENSE` (upstream MIT, unchanged).

**Deliberately excluded, differing from Hydrinia's tree** (Buddy/K1
decision, conservative default — see the build hand-back for the full
reasoning): Hydrinia additionally vendored `README.md`, `requirements.txt`,
and `tests/` (the upstream golden-regression suite + its CSV fixtures).
This build does not copy them:

- `README.md` / upstream `requirements.txt` are provenance-only, not
  imported by any code path — CLIO's own `requirements-kronos.txt` (repo
  root) is the actual pin surface for this repo.
- `tests/` (`test_kronos_regression.py` + `tests/data/*.csv`) is a
  network-and-weights-dependent golden-output check. Hydrinia's own
  `tests/conftest.py` skips it at COLLECTION time via a `snapshot_download`
  probe, but that probe itself imports `huggingface_hub` and, on a cache
  miss, would attempt a network call — this repo's test suite must run
  fully offline with a venv that has no `torch` installed at all (K1's
  build venv: no torch). Including the regression test would risk
  `import torch` reaching pytest collection on any machine that DOES have
  a `tests/` `conftest.py` bug or a warm HF cache without the exact pinned
  revision. Per the K1 build brief ("exclude ... unless a tiny golden test
  is needed"): no golden test is included in this build. If one is wanted
  later, copy `tests/test_kronos_regression.py` + `tests/data/` from
  Hydrinia's tree and give it its OWN `conftest.py` collect-guard that
  never imports `huggingface_hub`/`torch` at collection time (e.g. a bare
  `os.path.exists()` check on the local HF cache directory), so `pytest
  tests/` on a torch-less venv still collects cleanly.

### Required vendor patches (2, both in `model/kronos.py`) — copied from
Hydrinia's tree, already applied (verified: no `sys.path` manipulation
anywhere in `model/*.py`; `kronos.py` imports `.module` relatively)

1. `from model.module import *` → `from .module import *` (relative import,
   addressable as `agent.kronos.vendor.model` without `sys.path`
   injection).
2. Upstream's leftover `sys.path.append("../")` (and its `import sys`)
   removed — never widen `sys.path` at import inside a live service.

Both patches were already present in the source tree this was copied from
(Hydrinia's own record: line 1 applied 2026-06-10 by the kronos-p1 CC
session; line 2 applied 2026-06-10 during that session's verification
pass, SEC-F2). No further patching was needed for this copy.

## 3. Pinned weights (HF revisions — code-enforced, never re-resolved)

- `NeoQuasar/Kronos-small` → `901c26c1332695a2a8f243eb2f37243a37bea320`
  (matches the upstream regression test's `MODEL_REVISION` pin, per
  Hydrinia's record)
- `NeoQuasar/Kronos-Tokenizer-base` → `0e0117387f39004a9016484a186a908917e22426`

`agent/kronos/config.py`'s `KronosConfig.model_revision` /
`tokenizer_revision` defaults equal these exactly (`from_env()` can
override, but the shipped default is this pin). The service loads with
`local_files_only=True` always (never a supply-chain-open default); the
one exception is `python -m agent.kronos --download`, a one-time, explicit
deploy step that fetches exactly these two pinned revisions into the local
HF cache and then exits — it never fetches "latest".

## 4. Dependency surface

Same as Hydrinia's (`requirements-kronos.txt` at the CLIO repo root pins
CPU torch + pandas + einops + safetensors + huggingface_hub — see that
file for exact versions, chosen to match Hydrinia's verified-working prod
pins where sensible). `agent/kronos`'s own `requirements.txt` for the
vendor package is not duplicated here; the repo-root file is the single
pin surface (avoids the two-file drift CLAUDE.md warns about elsewhere in
this repo, e.g. DPOF's config split history).

## 5. Provisioning (deploy step, not done by this build)

This build (K1, worktree-only) does NOT run `--download` and does NOT have
`torch` in its venv (by design — the build/test venv must never require
torch). Fetching the pinned weights into the HF cache on the real VPS venv
is a deploy step:

```bash
cd /home/agent/openclaw
./venv/bin/pip install -r requirements-kronos.txt
./venv/bin/python3 -m agent.kronos --download
./venv/bin/python3 -m agent.kronos --once   # smoke test, writes files with --dry-run first if preferred
```

## 6. Verification run in this build

```bash
python3 -m py_compile agent/kronos/vendor/model/*.py
```

No import-level check of `agent.kronos.vendor.model` was possible in this
build's venv (no torch/einops/huggingface_hub installed there by design —
see the K1 build brief: "no torch installed — correct; nothing may require
it"). The isolation test (`tests/test_kronos_isolation.py`) instead proves
that importing `agent.kronos` (and its submodules other than the vendor
package / the predictor's lazy-load path) never imports `torch`.
