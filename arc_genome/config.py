"""Cumulative phase configuration for ARC-Genome."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseConfig:
    level: int = 0
    # Phase 1 — calibrated cost + gate
    calibrated_cost: bool = False
    cost_gate_min_score: float = 0.0
    # Phase 2 — cheap conv
    max_kernel: int = 29
    content_normalize: bool = False
    conv_sparsify: bool = False
    prefer_structural: bool = False
    # Phase 3 — analytical expansion
    extended_analytical: bool = False
    family_solvers: bool = False
    # Phase 4 — composition
    composition_depth: int = 0
    # Phase 5 — hard tasks
    conv_v2: bool = False
    unsolved_budget: float = 30.0
    unsolved_max_kernel: int = 9
    second_pass: bool = False
    # Phase 6 — audit + ARC-GEN
    arcgen_validation: bool = False
    cost_audit: bool = False


def build_config(level: int) -> PhaseConfig:
    cfg = PhaseConfig(level=level)
    if level >= 1:
        cfg.calibrated_cost = True
        cfg.cost_gate_min_score = 8.0
    if level >= 2:
        cfg.max_kernel = 11
        cfg.content_normalize = True
        cfg.conv_sparsify = True
        cfg.prefer_structural = True
    if level >= 3:
        cfg.extended_analytical = True
        cfg.family_solvers = True
    if level >= 4:
        cfg.composition_depth = 3
    if level >= 5:
        cfg.conv_v2 = True
        cfg.unsolved_budget = 90.0
        cfg.unsolved_max_kernel = 11
        cfg.second_pass = True
    if level >= 6:
        cfg.arcgen_validation = True
        cfg.cost_audit = True
    return cfg


_current = build_config(0)


def set_phase(level: int) -> PhaseConfig:
    global _current
    _current = build_config(level)
    return _current


def get_config() -> PhaseConfig:
    return _current
