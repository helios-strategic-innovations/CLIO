"""
Moltbot Memory Architecture — 5-Tier Hydrinia-Modelled System

Tier 1: Working Memory    — hot, sliding window, max 50 entries
Tier 2: Episodic Memory   — warm, append-only JSONL event log
Tier 3: Semantic Memory    — cold, structured knowledge (via knowledge_tools.py)
Tier 4: Procedural Memory  — persistent, learned behavioral patterns
Tier 5: Meta-Memory        — persistent, self-awareness and calibration

Usage:
    from agent.memory import get_memory_engine
    engine = get_memory_engine()
    engine.remember("key", "content", priority=0.8)
    context = engine.get_full_context(domain="trading")
"""

import os

# S162 L-6: Shared data directory constant — eliminates triple-dirname in 4 tier modules
MEMORY_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "memory",
)

from .memory_engine import MemoryEngine, get_memory_engine

__all__ = ["MemoryEngine", "get_memory_engine", "MEMORY_DATA_DIR"]
