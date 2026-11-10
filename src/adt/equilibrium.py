"""Window-based basal preparation; no integration is performed on import."""

from dataclasses import dataclass, replace
from math import isclose
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .inputs import ConstantInput, input_from_config
from .interventions import Intervention
from .model import rhs, state_vector
from .parameters import (
    ConfigDraft, ModelParameters, SubjectParameters, check_keys, finite_number,
    load_config, mapping,
)
from .simulate import SimulationError, SimulationSettings, settings_from_config, simulate
from .states import STATE_NAMES


def _steps(duration: float, interval: float, name: str) -> int:
    """Require complete, evenly sampled windows, without rounding a partial one."""
    ratio = duration / interval
    if not np.isfinite(ratio) or ratio < 1:
        raise ValueError(f"{name}: must contain at least one complete interval.")
    count = round(ratio)
    if not isclose(ratio, count, rel_tol=0., abs_tol=1e-9):
        raise ValueError(f"{name}: must be an integer multiple of its interval.")
    return count


@dataclass(frozen=True)
class PreparationSettings:
    """Stopping tolerances are distinct from the solver's local error controls."""

    profile: str
    minimum_time: float
    maximum_time: float
    window: float
    consecutive_windows: int
    check_interval: float
    derivative_guard_horizon: float
    state_absolute_tolerances: tuple[float, ...]
    state_relative_tolerance: float
    initial_state: tuple[float, ...]
    solver: SimulationSettings

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a nonempty label.")
        for name in ("maximum_time", "window", "check_interval",
                     "derivative_guard_horizon", "state_relative_tolerance"):
            value = finite_number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0 <= finite_number(self.minimum_time, "minimum_time") <= self.maximum_time:
            raise ValueError("minimum_time must be between zero and maximum_time.")
        if (isinstance(self.consecutive_windows, bool)
                or not isinstance(self.consecutive_windows, int)
                or self.consecutive_windows < 1):
            raise ValueError("consecutive_windows must be a positive integer.")
        windows = _steps(self.maximum_time, self.window, "maximum_time")
        points = _steps(self.window, self.check_interval, "window")
        if self.consecutive_windows > windows:
            raise ValueError("The time budget cannot contain the required windows.")
        absolute = tuple(finite_number(x, "state_absolute_tolerances")
                         for x in self.state_absolute_tolerances)
        if len(absolute) != len(STATE_NAMES) or any(x <= 0 for x in absolute):
            raise ValueError("Supply seven positive state_absolute_tolerances.")
        initial = tuple(float(x) for x in state_vector(self.initial_state))
        if self.solver.t_span != (0., self.window):
            raise ValueError("The solver template must describe the first complete window.")
        grid = np.linspace(0., self.window, points + 1)
        if not np.array_equal(self.solver.t_eval, grid):
            raise ValueError("Solver check times must match check_interval.")
        if self.solver.negative_tolerance != 0:
            raise ValueError("Basal preparation requires strict nonnegative states.")
        object.__setattr__(self, "state_absolute_tolerances", absolute)
        object.__setattr__(self, "initial_state", initial)


def preparation_from_config(draft: ConfigDraft) -> PreparationSettings:
    """Validate the dedicated preparation schema before any RHS or solver call."""
    metadata = ("time_unit", "state_normalization", "inflammation_normalization")
    keys = {
        "minimum_time", "maximum_time", "window", "consecutive_windows",
        "check_interval", "derivative_guard_horizon", "state_absolute_tolerances",
        "state_relative_tolerance",
    }
    check_keys(draft.data, {"profile", "preparation", "solver", "initial_state", *metadata},
               "preparation config")
    required = draft.require_values("profile", "solver", "initial_state", *metadata,
                                    *(f"preparation.{key}" for key in sorted(keys)))
    block = mapping(draft.data["preparation"], "preparation")
    check_keys(block, keys, "preparation")
    window = finite_number(block["window"], "window")
    interval = finite_number(block["check_interval"], "check_interval")
    if window <= 0 or interval <= 0:
        raise ValueError("window and check_interval must be positive.")
    count = _steps(window, interval, "window")
    solver_data = {name: required[name] for name in (*metadata, "solver", "initial_state")}
    solver_data.update(time_span={"start": 0., "end": window},
                       evaluation_times=np.linspace(0., window, count + 1).tolist())
    solver, initial = settings_from_config(ConfigDraft(draft.source, solver_data))
    absolute = block["state_absolute_tolerances"]
    if not isinstance(absolute, (list, tuple)):
        raise ValueError("state_absolute_tolerances must be a seven-value sequence.")
    return PreparationSettings(
        profile=required["profile"], **{k: v for k, v in block.items()
                                        if k != "state_absolute_tolerances"},
        state_absolute_tolerances=tuple(absolute), initial_state=tuple(initial), solver=solver,
    )


def load_preparation(path: str | Path) -> PreparationSettings:
    """Read complete controls without evaluating a scientific trajectory."""
    return preparation_from_config(load_config(path, allow_incomplete=False))


def basal_parameters(draft: ConfigDraft) -> ModelParameters:
    """Require the basal reference conditions; unused GSM settings are irrelevant."""
    parameters = ModelParameters.from_config(draft)
    subject = SubjectParameters.from_config(draft)
    forcing = input_from_config(draft)
    if subject.C_i != 1 or not isinstance(forcing, ConstantInput) or forcing.I_ext != 0:
        raise ValueError("Basal preparation requires C_i=1 and constant I_ext=0.")
    return parameters


@dataclass(frozen=True)
class WindowCheck:
    start: float
    end: float
    residual: float
    variation: float
    residual_by_state: tuple[float, ...]
    variation_by_state: tuple[float, ...]
    worst_residual_state: str
    worst_residual_time: float
    worst_variation_state: str
    passes: bool
    consecutive: int = 0


def assess_window(
    times: NDArray[np.float64], states: NDArray[np.float64],
    derivatives: NDArray[np.float64], settings: PreparationSettings,
) -> WindowCheck:
    """Check all seven states throughout a sampled window, including near-zero pools."""
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
            or states.shape != (len(STATE_NAMES), len(times))
            or derivatives.shape != states.shape):
        raise ValueError("Invalid window dimensions or check times.")
    if (not np.all(np.isfinite(states)) or np.any(states < 0)
            or not np.all(np.isfinite(derivatives))):
        raise ValueError("Window contains negative/nonfinite states or nonfinite derivatives.")
    scale = (np.asarray(settings.state_absolute_tolerances)
             + settings.state_relative_tolerance * np.max(np.abs(states), axis=1))
    scaled = settings.derivative_guard_horizon * np.abs(derivatives) / scale[:, None]
    variation = np.ptp(states, axis=1) / scale
    if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(variation)):
        raise ValueError("Nonfinite convergence metrics.")
    state_index, time_index = np.unravel_index(np.argmax(scaled), scaled.shape)
    residual = float(np.max(scaled))
    drift = float(np.max(variation))
    return WindowCheck(
        start=float(times[0]), end=float(times[-1]), residual=residual, variation=drift,
        residual_by_state=tuple(float(x) for x in np.max(scaled, axis=1)),
        variation_by_state=tuple(float(x) for x in variation),
        worst_residual_state=STATE_NAMES[state_index], worst_residual_time=float(times[time_index]),
        worst_variation_state=STATE_NAMES[int(np.argmax(variation))],
        passes=residual <= 1 and drift <= 1,
    )


@dataclass
class PreparationResult:
    status: Literal["converged", "budget_exhausted", "numerical_failure"]
    message: str
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    windows: list[WindowCheck]
    nfev: int
    njev: int
    nlu: int
    attempted_span: tuple[float, float] | None = None


def prepare_reference(parameters: ModelParameters, settings: PreparationSettings) -> PreparationResult:
    """Integrate basal windows; an unsuccessful last state is diagnostic only."""
    subject, forcing, control = SubjectParameters(1.), ConstantInput(0.), Intervention("control")
    current = state_vector(settings.initial_state).copy()
    time_blocks = [np.array([0.])]
    state_blocks = [current[:, None].copy()]
    checks: list[WindowCheck] = []
    consecutive = nfev = njev = nlu = 0
    status = "budget_exhausted"
    message = "Preparation time budget exhausted without operational convergence."
    attempted_span = None
    count = _steps(settings.window, settings.check_interval, "window")
    for index in range(_steps(settings.maximum_time, settings.window, "maximum_time")):
        left, right = index * settings.window, (index + 1) * settings.window
        solver = replace(settings.solver, t_span=(left, right),
                         t_eval=tuple(np.linspace(left, right, count + 1)))
        try:
            result = simulate(parameters, subject, forcing, control, current, solver)
            nfev += result.nfev
            njev += result.njev
            nlu += result.nlu
            if not np.array_equal(result.y[:, 0], current):
                raise ValueError("Window did not preserve its initial state.")
            derivatives = np.column_stack([
                rhs(t, x, parameters, subject, forcing, control)
                for t, x in zip(result.t, result.y.T)
            ])
            check = assess_window(result.t, result.y, derivatives, settings)
        except (SimulationError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            status, message = "numerical_failure", f"{type(error).__name__}: {error}"
            attempted_span = (left, right)
            break
        consecutive = consecutive + 1 if check.passes else 0
        checks.append(replace(check, consecutive=consecutive))
        time_blocks.append(result.t[1:])
        state_blocks.append(result.y[:, 1:])
        current = result.y[:, -1].copy()
        if consecutive >= settings.consecutive_windows and right >= settings.minimum_time:
            status, message = "converged", "Seven-state window criteria satisfied; stage 06 pending."
            break
    return PreparationResult(status, message, np.concatenate(time_blocks),
                             np.concatenate(state_blocks, axis=1), checks,
                             nfev, njev, nlu, attempted_span)
