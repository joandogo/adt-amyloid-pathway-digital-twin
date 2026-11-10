"""Explicit stage 07 plan validation and separate-axis expansion; no RHS calls."""

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
import re

from .inputs import ConstantInput
from .outcomes import AgreementSettings
from .parameters import ConfigDraft, ModelParameters, SubjectParameters, check_keys, finite_number, mapping
from .reference_artifacts import ArchivedReference
from .simulate import SimulationSettings
from .states import STATE_NAMES


OUTCOME_SPEC = {
    "auc_states": ["A42", "O", "I", "C99"],
    "final_states": list(STATE_NAMES), "maximum_states": ["C99"],
    "ratio": "A42_over_A40_at_checkpoints",
    "contrast": "difference_from_time_matched_unperturbed_control",
    "threshold_time_enabled": False, "biological_classification": "unclassified",
}


def _section(draft: ConfigDraft, path: str, keys: set[str]) -> dict[str, object]:
    draft.require_values(*(f"{path}.{key}" for key in sorted(keys)))
    current = draft.data
    for part in path.split("."):
        current = mapping(current[part], path)
    check_keys(current, keys, path)
    return current


def _numbers(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name}: a nonempty sequence is required.")
    return tuple(finite_number(item, name) for item in value)


def make_grid(segments: object, horizon: float, checkpoints: tuple[float, ...]) -> tuple[float, ...]:
    """Tile decimal segments, include checkpoints and retain exact shared samples."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("output_grid requires explicit segments.")
    points: list[float] = []
    last = Decimal(0)
    for item in segments:
        segment = mapping(item, "output_grid segment")
        check_keys(segment, {"start", "end", "step"}, "output_grid segment")
        if set(segment) != {"start", "end", "step"}:
            raise ValueError("Each grid segment requires start, end and step.")
        start, end, step = (Decimal(str(finite_number(segment[key], f"grid.{key}")))
                            for key in ("start", "end", "step"))
        if start != last or end <= start or step <= 0:
            raise ValueError("Grid segments must tile [0,horizon] with positive steps.")
        count = (end - start) / step
        if count != count.to_integral_value() or count > 1_000_000:
            raise ValueError("A grid segment must have an integral, at most 1000000, step count.")
        points.extend(float(start + j * step) for j in range(int(count)))
        last = end
    if last != Decimal(str(horizon)):
        raise ValueError("The output grid must end at the common horizon.")
    grid = tuple(sorted(set((*points, horizon, *checkpoints))))
    if len(grid) < 2 or grid[0] != 0 or grid[-1] != horizon:
        raise ValueError("Invalid common output grid.")
    return grid


def refine_grid(times: tuple[float, ...]) -> tuple[float, ...]:
    """Bisect actual intervals; coarse samples remain bit-for-bit identical."""
    refined: list[float] = []
    for left, right in zip(times, times[1:]):
        middle = left + (right - left) / 2
        if not left < middle < right:
            raise ValueError("Grid is too fine for distinct midpoint samples.")
        refined.extend((left, middle))
    return (*refined, times[-1])


@dataclass(frozen=True)
class SweepPlan:
    profile: str
    reference: dict[str, str]
    levels: dict[str, tuple[float, ...]]
    horizon: float
    checkpoints: tuple[float, ...]
    times: tuple[float, ...]
    refined_times: tuple[float, ...]
    solver: dict[str, object]
    refinement: dict[str, object]
    agreement: AgreementSettings
    output_root: Path


def plan_from_config(draft: ConfigDraft) -> SweepPlan:
    top = {"profile", "status", "reference", "state_order", "intervention", "design",
           "evaluation", "numerics", "outcomes", "output_root"}
    check_keys(draft.data, top, "sweeps")
    draft.require_values(*sorted(top))
    data = draft.data
    if not isinstance(data["profile"], str) or not data["profile"].strip():
        raise ValueError("A named sweep profile is required.")
    if data["status"] != "selected":
        raise ValueError("The sweep design must be selected before use.")
    if data["state_order"] != list(STATE_NAMES) or data["intervention"] != "control":
        raise ValueError("Stage 07 requires the v0.3 state order and control intervention.")
    reference = _section(draft, "reference", {
        "preparation_dir", "confirmed_file", "confirmed_sha256", "parameter_source"})
    if any(not isinstance(x, str) or not x.strip() for x in reference.values()):
        raise ValueError("Reference fields must be nonempty strings.")
    if reference["parameter_source"] != "archived_preparation_snapshot":
        raise ValueError("Use the archived preparation parameters, not current baseline values.")
    if re.fullmatch(r"[0-9a-f]{64}", reference["confirmed_sha256"]) is None:
        raise ValueError("confirmed_sha256 must be an explicit lowercase SHA-256 digest.")
    design = _section(draft, "design", {
        "protocol", "k_BACE_multipliers", "C_i_values", "I_ext_values"})
    if design["protocol"] != "common_basal_step":
        raise ValueError("Only the selected common_basal_step protocol is implemented.")
    levels = {axis: _numbers(design[key], key) for axis, key in
              (("k_BACE", "k_BACE_multipliers"), ("C_i", "C_i_values"), ("I_ext", "I_ext_values"))}
    for axis, values in levels.items():
        nominal = 0. if axis == "I_ext" else 1.
        if len(values) < 2 or len(set(values)) != len(values) or nominal not in values:
            raise ValueError(f"{axis}: unique levels must include nominal and a changed level.")
        if (axis == "k_BACE" and min(values) < 1 or axis == "C_i" and not 0 < min(values) <= max(values) <= 1
                or axis == "I_ext" and min(values) < 0):
            raise ValueError(f"{axis}: levels violate the selected causal-driver direction.")
    evaluation = _section(draft, "evaluation", {"start", "horizon", "checkpoints", "output_grid"})
    if finite_number(evaluation["start"], "evaluation.start") != 0:
        raise ValueError("Local evaluation starts at tau=0 from the full confirmed state.")
    horizon = finite_number(evaluation["horizon"], "horizon")
    checkpoints = _numbers(evaluation["checkpoints"], "checkpoints")
    if (horizon <= 0 or checkpoints[0] <= 0 or checkpoints[-1] != horizon
            or any(b <= a for a, b in zip(checkpoints, checkpoints[1:]))):
        raise ValueError("Checkpoints must increase in (0,horizon] and include the horizon.")
    times = make_grid(evaluation["output_grid"], horizon, checkpoints)
    _section(draft, "numerics", {"solver", "refinement", "agreement"})
    solver = _section(draft, "numerics.solver", {"method", "rtol", "atol", "max_step", "negative_tolerance"})
    # Validate independently of reference loading, without invoking RHS.
    SimulationSettings((0., horizon), times, **solver, time_unit="pending archived metadata",
                       state_normalization="pending archived metadata", inflammation_normalization="pending archived metadata")
    if solver["method"] != "Radau" or finite_number(solver["negative_tolerance"], "negative_tolerance") != 0:
        raise ValueError("U1 uses Radau and strict nonnegativity.")
    refinement = _section(draft, "numerics.refinement", {
        "tolerance_factor", "max_step_factor", "grid_subdivisions", "comparison_method"})
    for key in ("tolerance_factor", "max_step_factor"):
        if not 0 < finite_number(refinement[key], key) < 1:
            raise ValueError(f"{key} must lie strictly between zero and one.")
    if type(refinement["grid_subdivisions"]) is not int or refinement["grid_subdivisions"] != 2:
        raise ValueError("The selected grid refinement bisects every interval.")
    if refinement["comparison_method"] != "BDF":
        raise ValueError("U1 compares refined Radau with refined BDF.")
    agreement = AgreementSettings(**_section(draft, "numerics.agreement", {
        "absolute_tolerances", "relative_tolerance", "ratio_absolute_tolerance"}))
    outcomes = _section(draft, "outcomes", set(OUTCOME_SPEC))
    if outcomes != OUTCOME_SPEC or outcomes["threshold_time_enabled"] is not False:
        raise ValueError("Unsupported stage 07 outcome specification; no threshold classification is implemented.")
    if not isinstance(data["output_root"], str) or not data["output_root"].strip():
        raise ValueError("output_root must be an explicit path.")
    return SweepPlan(data["profile"], dict(reference), levels, horizon, checkpoints, times,
                     refine_grid(times), dict(solver), dict(refinement), agreement, Path(data["output_root"]))


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    parameters: ModelParameters
    subject: SubjectParameters
    external_input: ConstantInput


def build_cases(plan: SweepPlan, nominal: ModelParameters) -> tuple[dict[str, SweepCase], list[dict[str, object]]]:
    """Share one baseline; never expand the Cartesian product of driver levels."""
    cases = {"baseline": SweepCase("baseline", replace(nominal), SubjectParameters(1.), ConstantInput(0.))}
    membership: list[dict[str, object]] = []
    for axis, levels in plan.levels.items():
        for index, level in enumerate(levels):
            is_baseline = level == (0. if axis == "I_ext" else 1.)
            name = "baseline" if is_baseline else f"{axis}_{index:03d}"
            if not is_baseline:
                parameters = replace(nominal, k_BACE=nominal.k_BACE * level) if axis == "k_BACE" else replace(nominal)
                cases[name] = SweepCase(name, parameters,
                    SubjectParameters(level if axis == "C_i" else 1.),
                    ConstantInput(level if axis == "I_ext" else 0.))
            membership.append({"axis": axis, "level": level, "case_id": name,
                               "level_kind": "reference_multiplier" if axis == "k_BACE" else "absolute"})
    return cases, membership


def build_settings(plan: SweepPlan, source: ArchivedReference) -> dict[str, SimulationSettings]:
    """Carry archived coordinate descriptions unchanged into every solver variant."""
    archived = source.preparation.solver
    main = SimulationSettings((0., plan.horizon), plan.times, **plan.solver,
        time_unit=archived.time_unit, state_normalization=archived.state_normalization,
        inflammation_normalization=archived.inflammation_normalization)
    factor = plan.refinement["tolerance_factor"]
    atol = tuple(x * factor for x in main.atol) if isinstance(main.atol, tuple) else main.atol * factor
    refined = replace(main, t_eval=plan.refined_times, rtol=main.rtol * factor, atol=atol,
                      max_step=main.max_step * plan.refinement["max_step_factor"])
    return {"main": main, "radau_refined": refined,
            "bdf_refined": replace(refined, method=plan.refinement["comparison_method"])}
