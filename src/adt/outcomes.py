"""Checkpoint outcomes and explicit numerical agreement in normalized coordinates."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .parameters import finite_number
from .states import STATE_NAMES


@dataclass(frozen=True)
class AgreementSettings:
    absolute_tolerances: tuple[float, ...]
    relative_tolerance: float
    ratio_absolute_tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.absolute_tolerances, (list, tuple)):
            raise ValueError("Agreement absolute_tolerances must be a sequence.")
        absolute = tuple(finite_number(x, "absolute_tolerances") for x in self.absolute_tolerances)
        if len(absolute) != 7 or min(absolute) <= 0:
            raise ValueError("Agreement needs seven positive absolute tolerances.")
        object.__setattr__(self, "absolute_tolerances", absolute)
        for key in ("relative_tolerance", "ratio_absolute_tolerance"):
            value = finite_number(getattr(self, key), key)
            if value <= 0:
                raise ValueError(f"{key} must be positive.")
            object.__setattr__(self, key, value)


def validate_trajectory(times: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require complete, ordered, finite, nonnegative seven-state samples."""
    t, y = np.asarray(times), np.asarray(states)
    if t.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise ValueError("Trajectory samples must be real numbers.")
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or y.shape != (7, len(t)) or t[0] != 0
            or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0)
            or not np.all(np.isfinite(y)) or np.any(y < 0)):
        raise ValueError("Invalid complete seven-state trajectory starting at zero.")
    return t, y


def at_times(times: np.ndarray, states: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Select actual samples, rejecting missing coverage instead of interpolating."""
    t, y = validate_trajectory(times, states)
    requested = np.asarray(requested, dtype=float)
    if requested.ndim != 1 or not len(requested) or not np.all(np.isfinite(requested)):
        raise ValueError("Requested times must be a finite nonempty vector.")
    indices = np.searchsorted(t, requested)
    if np.any(indices >= len(t)) or not np.array_equal(t[indices], requested):
        raise ValueError("Trajectory lacks the complete requested common samples.")
    return y[:, indices]


def evaluate_outcomes(times: np.ndarray, states: np.ndarray, checkpoints: tuple[float, ...]) -> list[dict[str, object]]:
    """Trapezoidal AUC, sampled maxima, endpoints and endpoint A42/A40 ratios."""
    t, y = validate_trajectory(times, states)
    rows: list[dict[str, object]] = []
    for raw_h in checkpoints:
        h = finite_number(raw_h, "checkpoint")
        index = int(np.searchsorted(t, h))
        if h <= 0 or index >= len(t) or t[index] != h:
            raise ValueError("Each positive checkpoint must be an actual trajectory sample.")
        def append(metric: str, value: float | None, reason: str | None = None,
                   sample_time: float | None = None) -> None:
            if value is not None and not isfinite(value):
                value, reason = None, "nonfinite_outcome"
            rows.append({"checkpoint": h, "metric": metric, "value": value,
                         "reason": reason, "sample_time": sample_time})
        for j, name in enumerate(STATE_NAMES):
            append(f"{name}_final", float(y[j, index]))
        for name in ("A42", "O", "I", "C99"):
            values = y[STATE_NAMES.index(name), :index + 1]
            # Direct trapezoidal definition, compatible with existing NumPy versions.
            with np.errstate(over="ignore", invalid="ignore"):
                area = float(np.sum((values[:-1] / 2 + values[1:] / 2) * np.diff(t[:index + 1])))
            append(f"AUC_{name}", area)
        maximum_index = int(np.argmax(y[STATE_NAMES.index("C99"), :index + 1]))
        append("C99_max", float(y[1, maximum_index]), sample_time=float(t[maximum_index]))
        denominator = float(y[2, index])
        if denominator == 0:
            append("Ratio_42_40", None, "A40_zero_at_checkpoint")
        else:
            append("Ratio_42_40", float(y[3, index]) / denominator)
    return rows


def compare_trajectories(left: np.ndarray, right: np.ndarray, settings: AgreementSettings) -> dict[str, object]:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 7 or left.size == 0:
        raise ValueError("Compare matching nonempty seven-state trajectories.")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (left, right)):
        raise ValueError("Agreement requires finite nonnegative trajectories.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = np.abs(left - right) / (np.asarray(settings.absolute_tolerances)[:, None]
                    + settings.relative_tolerance * np.maximum(np.abs(left), np.abs(right)))
    if not np.all(np.isfinite(difference)):
        raise ValueError("Nonfinite trajectory agreement metric.")
    per_state = np.max(difference, axis=1)
    return {"passes": bool(np.max(per_state) <= 1), "maximum": float(np.max(per_state)),
            "by_state": dict(zip(STATE_NAMES, map(float, per_state))),
            "worst_state": STATE_NAMES[int(np.argmax(per_state))]}


def _index_rows(rows: list[dict[str, object]]) -> dict[tuple[float, str], dict[str, object]]:
    indexed = {(row["checkpoint"], row["metric"]): row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError("Outcome rows must be nonempty and unique by checkpoint/metric.")
    return indexed


def _absolute_tolerance(metric: str, checkpoint: float, settings: AgreementSettings) -> float:
    if metric == "Ratio_42_40":
        return settings.ratio_absolute_tolerance
    state = metric[4:] if metric.startswith("AUC_") else metric.rsplit("_", 1)[0]
    absolute = settings.absolute_tolerances[STATE_NAMES.index(state)]
    return absolute * checkpoint if metric.startswith("AUC_") else absolute


def compare_outcomes(left: list[dict[str, object]], right: list[dict[str, object]],
                     settings: AgreementSettings) -> dict[str, object]:
    """AUC absolute tolerances scale with interval length; ratios have their own."""
    left_index, right_index = _index_rows(left), _index_rows(right)
    if left_index.keys() != right_index.keys():
        raise ValueError("Outcome comparisons require identical checkpoint/metric coverage.")
    checks: list[dict[str, object]] = []
    for key, row in left_index.items():
        other = right_index[key]
        a, b = row["value"], other["value"]
        absolute = _absolute_tolerance(key[1], key[0], settings)
        scaled = None
        if a is None or b is None:
            # Both zero-denominator ratios are undefined, not zero and not numerical evidence.
            unavailable = (key[1] == "Ratio_42_40" and a is None and b is None
                           and row["reason"] == other["reason"] == "A40_zero_at_checkpoint")
            passes = unavailable
            status = "not_applicable" if unavailable else "undefined_or_nonfinite_mismatch"
        else:
            a, b = finite_number(a, "outcome"), finite_number(b, "outcome")
            scaled = abs(a - b) / (absolute + settings.relative_tolerance * max(abs(a), abs(b)))
            if not isfinite(scaled):
                scaled, passes, status = None, False, "nonfinite_agreement"
            else:
                passes = scaled <= 1
                status = "agrees" if passes else "disagrees"
        checks.append({"checkpoint": key[0], "metric": key[1], "passes": passes,
                       "status": status, "scaled_difference": scaled, "absolute_tolerance": absolute})
    values = [row["scaled_difference"] for row in checks if row["scaled_difference"] is not None]
    return {"passes": all(row["passes"] for row in checks), "maximum": max(values) if values else None,
            "not_applicable_count": sum(row["status"] == "not_applicable" for row in checks), "checks": checks}


def contrast_outcomes(rows: list[dict[str, object]], control: list[dict[str, object]] | None) -> list[dict[str, object]]:
    """Compare a driver arm with the full time-matched unperturbed trajectory."""
    baseline = _index_rows(control) if control is not None else {}
    if control is not None and _index_rows(rows).keys() != baseline.keys():
        raise ValueError("Control outcomes must cover the same checkpoints and metrics.")
    output = []
    for row in rows:
        reference = baseline.get((row["checkpoint"], row["metric"]))
        value = row["value"]
        control_value = None if reference is None else reference["value"]
        difference = percent = None
        reason = None
        if reference is None:
            reason = "control_run_unavailable"
        elif value is None or control_value is None:
            reason = "outcome_or_control_undefined"
        else:
            difference = float(value - control_value)
            if not isfinite(difference):
                difference, reason = None, "nonfinite_difference"
            elif control_value == 0:
                reason = "control_denominator_zero"
            else:
                percent = 100 * (difference / control_value)
                if not isfinite(percent):
                    percent, reason = None, "nonfinite_percent_change"
        output.append({**row, "control_value": control_value, "difference_from_control": difference,
                       "percent_change_from_control": percent, "contrast_reason": reason})
    return output
