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
    arcgen_validate_samples: int = 30
    arcgen_fit_samples: int = 50
    use_official_score: bool = False
    third_pass: bool = False
    third_pass_budget: float = 60.0
    # Phase 10 — M5 variable-shape + object primitives
    milestone5_solvers: bool = False
    # Phase 11 — bounded code world models (dynamic bbox transforms)
    bounded_world: bool = False
    # Phase 13 — ARC-GEN signature routing (M8)
    arcgen_routing: bool = False
    # Phase 14 — tile-based bounded upscale (cheaper memory)
    bounded_tile_upscale: bool = False
    # Phase 15 — ARC-GEN-fitted gather indices
    arcgen_fit_gather: bool = False
    # Phase 16 — ARC-GEN-fitted color remaps
    arcgen_fit_color: bool = False
    # Phase 17 — bbox-relative ARC-GEN-fitted gather
    arcgen_fit_bbox_gather: bool = False
    # Phase 18 — ARC-GEN-fitted object programs (gravity, component, rect rules)
    arcgen_fit_object_programs: bool = False
    # Phase 19 — ARC-GEN extract/place (bbox shift, patch paste, translate)
    arcgen_fit_place_programs: bool = False
    # Phase 20 — ARC-GEN depth-2 composition (color ∘ flip, gravity ∘ map, …)
    arcgen_compose_depth: int = 0
    # Phase 21 — dynamic bbox-relative gravity ONNX (variable-shape safe)
    arcgen_dynamic_gravity: bool = False


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
        cfg.arcgen_validate_samples = 30
    if level >= 7:
        cfg.arcgen_validate_samples = 100
    if level >= 8:
        cfg.use_official_score = True
        cfg.arcgen_fit_samples = 100
    if level >= 9:
        cfg.composition_depth = 4
        cfg.third_pass = True
        cfg.third_pass_budget = 60.0
        cfg.unsolved_budget = 90.0
    if level >= 10:
        cfg.milestone5_solvers = True
    if level >= 11:
        cfg.bounded_world = True
    if level >= 12:
        cfg.bounded_world = True  # slim flip compiler (default in bounded.py)
    if level >= 13:
        cfg.arcgen_routing = True
    if level >= 14:
        cfg.bounded_tile_upscale = True
        cfg.third_pass = False  # saturated at 0 wins; save ~30min
    if level >= 15:
        cfg.arcgen_fit_gather = True
    if level >= 16:
        cfg.arcgen_fit_color = True
    if level >= 17:
        cfg.arcgen_fit_bbox_gather = True
    if level >= 18:
        cfg.arcgen_fit_object_programs = True
    if level >= 19:
        cfg.arcgen_fit_place_programs = True
    if level >= 20:
        cfg.arcgen_compose_depth = 2
    if level >= 21:
        cfg.arcgen_dynamic_gravity = True
    if level >= 22:
        cfg.arcgen_compose_depth = 3
    return cfg


_current = build_config(0)


def set_phase(level: int) -> PhaseConfig:
    global _current
    _current = build_config(level)
    return _current


def get_config() -> PhaseConfig:
    return _current
