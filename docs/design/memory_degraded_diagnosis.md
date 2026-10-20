# "Self-check: memory degraded tiers [episodic]" — diagnosis (S215 / R-32c)

Status: diagnosis only. No code fix applied.

## Source

- Emitted by `run_self_check()` in `agent/integrations/scheduled_tasks.py` (~L556), WARNING level,
  when `MemoryEngine.health_check()` (`agent/memory/memory_engine.py` ~L845) reports
  `overall_healthy == False`. Only the episodic tier can make that flag false.

## Condition

`episodic_stale` starts True and is cleared only when `episodic.get_recent(limit=1)` returns a record
with a `timestamp` less than 24 h old. It also stays True if the store is empty or the read or parse
raises. The self-check runs every 10 minutes.

## Evidence (read-only)

- `journalctl -u moltbot.service`: 138 WARNING lines in two contiguous runs at 10-minute cadence:
  2026-10-09 13:48 to 2026-10-10 09:38 ET (120 lines), and 2026-10-11 09:50 to 12:40 ET (18 lines).
  The brief's "68 times from 10-10 01:28" does not match the journal.
- The episodic log is sparse: events are written on promote-on-expire and selected outcomes. Recent
  writes: 2026-10-08 17:39 UTC, 10-10 13:48 UTC, 10-11 16:50 UTC, 10-12 14:03 and 14:08 UTC.
- Each run starts about 24 h after the last write and ends at the next one. For example, the write at
  10-10 13:48 UTC (09:48 ET) ends run 1, and the write at 10-11 16:50 UTC (12:50 ET) ends run 2.
  The "stopped" in the brief is that next write, not a fix.

## Hypothesis

Expected behaviour on a sparse log, not a storage fault. Any gap over 24 h flips `overall_healthy`,
and the WARNING repeats every 10 minutes for the whole gap.

## Options (not applied)

1. Treat episodic staleness as informational (INFO, or drop it from `overall_healthy`).
2. Keep the signal but warn once per stale episode.
3. Keep as is if the episodic writer should be watched as a liveness check.

Operator decision needed: is episodic silence a health signal?
