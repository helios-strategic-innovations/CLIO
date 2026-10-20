"""Vendored Kronos foundation model (github.com/shiyu-coder/Kronos, MIT).

Copied from Hydrinia's already-provisioned `kronos_vendor/` (see
`VENDOR_PIN.md` in this directory for the full provenance record — same
upstream pin, same two patches, same pinned HF weight revisions). If
`model/` is ever absent (fresh clone before the model files are copied in,
or a stripped deploy), `agent.kronos.forecaster`'s lazy predictor load maps
the resulting ImportError to a clean "kronos model unavailable" status —
never a crash.
"""
