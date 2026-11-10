"""Run selected one-dimensional sweeps and verify every distinct driver regime."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .confirmed_reference import ConfirmedReference
from .interventions import Intervention
from .outcomes import (at_times, compare_outcomes, compare_trajectories, contrast_outcomes,
                       evaluate_outcomes, validate_trajectory)
from .simulate import SimulationError, SimulationResult, SimulationSettings, simulate
from .states import STATE_NAMES
from .sweep_plan import SweepCase, SweepPlan, build_cases, build_settings


@dataclass
class SweepRun:
    status: str
    message: str
    settings: SimulationSettings
    result: SimulationResult | None
    outcomes: list[dict[str, object]]


@dataclass
class SweepExecution:
    summary: dict[str, object]
    runs: dict[str, dict[str, SweepRun]]
    outcomes: list[dict[str, object]]
    membership: list[dict[str, object]]


def _run(case: SweepCase, initial: np.ndarray, settings: SimulationSettings,
         checkpoints: tuple[float, ...]) -> SweepRun:
    result = None
    try:
        result = simulate(case.parameters, case.subject, case.external_input,
                          Intervention("control"), initial.copy(), settings)
        t, y = validate_trajectory(result.t, result.y)
        if (not result.success or result.state_names != STATE_NAMES
                or not np.array_equal(t, settings.t_eval) or not np.array_equal(y[:, 0], initial)):
            raise SimulationError("Result must preserve the complete requested grid and seven-state initial condition.")
        outcomes = evaluate_outcomes(t, y, checkpoints)
        return SweepRun("completed", result.message, settings, result, outcomes)
    except (SimulationError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        return SweepRun("failed", f"{type(error).__name__}: {error}", settings, result, [])


def _compare(left: SweepRun, right: SweepRun, plan: SweepPlan) -> dict[str, object]:
    if left.status != "completed" or right.status != "completed":
        return {"passes": False, "reason": "run_unavailable"}
    try:
        # Right must include every actual sample from the left grid.
        comparison = compare_trajectories(left.result.y,
            at_times(right.result.t, right.result.y, left.result.t), plan.agreement)
        outcomes = compare_outcomes(left.outcomes, right.outcomes, plan.agreement)
        return {"passes": comparison["passes"] and outcomes["passes"],
                "trajectory": comparison, "outcomes": outcomes}
    except (ValueError, ArithmeticError) as error:
        return {"passes": False, "reason": str(error)}


def _missing_outcomes(plan: SweepPlan) -> list[dict[str, object]]:
    metrics = [*(f"{name}_final" for name in STATE_NAMES),
               *(f"AUC_{name}" for name in ("A42", "O", "I", "C99")), "C99_max", "Ratio_42_40"]
    return [{"checkpoint": h, "metric": metric, "value": None,
             "reason": "primary_run_failed", "sample_time": None}
            for h in plan.checkpoints for metric in metrics]


def run_sweeps(plan: SweepPlan, reference: ConfirmedReference,
               on_run: Callable[[str, str, SweepRun], None] | None = None,
               on_start: Callable[[str, str], None] | None = None) -> SweepExecution:
    """Keep failures visible; main results are never replaced by a favorable solver."""
    source = reference.archive
    cases, membership = build_cases(plan, source.parameters)
    settings = build_settings(plan, source)
    initial = source.state.copy()
    runs: dict[str, dict[str, SweepRun]] = {}
    reports: dict[str, dict[str, object]] = {}
    issues: list[str] = []
    for name, case in cases.items():
        runs[name] = {}
        local_issues: list[str] = []
        for variant, controls in settings.items():
            if on_start is not None:
                on_start(name, variant)
            run = _run(case, initial, controls, plan.checkpoints)
            runs[name][variant] = run
            if on_run is not None:
                on_run(name, variant, run)
            if run.status != "completed":
                local_issues.append(f"{variant}: {run.message}")
        comparisons = {
            "main_vs_radau_refined": _compare(runs[name]["main"], runs[name]["radau_refined"], plan),
            "refined_methods": _compare(runs[name]["radau_refined"], runs[name]["bdf_refined"], plan),
        }
        for label, comparison in comparisons.items():
            if not comparison["passes"]:
                local_issues.append(f"{label}: numerical_agreement_failed")
        reports[name] = {"numerically_confirmed": not local_issues, "issues": local_issues,
                         "runs": {key: {"status": run.status, "message": run.message}
                                  for key, run in runs[name].items()}, "comparisons": comparisons}
        issues.extend(f"{name}: {issue}" for issue in local_issues)
    # Baseline is a genuine trajectory with exactly the same elapsed horizon.
    baseline_run = runs["baseline"]["main"]
    control = baseline_run.outcomes if baseline_run.status == "completed" else None
    rows: list[dict[str, object]] = []
    for name, variants in runs.items():
        primary = variants["main"]
        values = primary.outcomes if primary.status == "completed" else _missing_outcomes(plan)
        for row in contrast_outcomes(values, control):
            rows.append({"case_id": name, "primary_run_status": primary.status,
                "numerically_confirmed": reports[name]["numerically_confirmed"],
                "contrast_sources_numerically_confirmed": bool(reports[name]["numerically_confirmed"]
                    and reports["baseline"]["numerically_confirmed"]), **row})
    confirmed = not issues
    summary = {"status": "confirmed" if confirmed else "not_confirmed", "numerically_confirmed": confirmed,
        "biological_status": "unclassified", "profile": plan.profile, "protocol": "common_basal_step",
        "scope": "selected univariate driver levels, horizon, sampled trajectories and defined checkpoint outcomes only",
        "unique_cases": len(cases), "axis_positions": len(membership), "required_runs": len(cases) * len(settings),
        "horizon": plan.horizon, "checkpoints": plan.checkpoints, "issues": issues,
        "cases": reports, "undefined_primary_outcomes": sum(row["value"] is None for row in rows),
        "undefined_primary_ratios": sum(row["metric"] == "Ratio_42_40" and row["value"] is None for row in rows),
        "primary_variant": "main", "equilibrium_at_horizon_claimed": False}
    return SweepExecution(summary, runs, rows, membership)
