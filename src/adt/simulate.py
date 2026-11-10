"""Explicit, reusable solve_ivp interface; nothing runs on import."""

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .inputs import ConstantInput, PiecewiseConstantInput, input_from_config, input_required_paths
from .interventions import Intervention
from .model import rhs, state_vector
from .parameters import (
    ConfigDraft, IncompleteConfigurationError, ModelParameters, SubjectParameters, check_keys,
    finite_number, mapping, validate_baseline,
)
from .states import STATE_NAMES


SIMULATION_REQUIRED_PATHS: tuple[str, ...] = (
    "time_unit", "state_normalization", "inflammation_normalization",
    "time_span.start", "time_span.end", "evaluation_times",
    "solver.method", "solver.rtol", "solver.atol", "solver.max_step",
    "solver.negative_tolerance", *(f"initial_state.{name}" for name in STATE_NAMES),
)


@dataclass(frozen=True)
class SimulationSettings:
    """Explicit time grid and numerical choices, without scientific defaults."""

    t_span: tuple[float, float]
    t_eval: tuple[float, ...]
    method: str
    rtol: float
    atol: float | tuple[float, ...]
    max_step: float
    negative_tolerance: float
    time_unit: str
    state_normalization: str
    inflammation_normalization: str

    def __post_init__(self) -> None:
        span = tuple(finite_number(t, "time_span") for t in self.t_span)
        grid = tuple(finite_number(t, "evaluation_times") for t in self.t_eval)
        if len(span) != 2 or span[1] <= span[0]:
            raise ValueError("time_span must contain increasing start/end times.")
        if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])):
            raise ValueError("evaluation_times must contain at least two increasing times.")
        if grid[0] != span[0] or grid[-1] != span[1]:
            raise ValueError("evaluation_times must include both time_span endpoints exactly.")
        if self.method not in ("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"):
            raise ValueError("Unsupported solve_ivp method.")
        for name in ("rtol", "max_step"):
            if finite_number(getattr(self, name), f"solver.{name}") <= 0:
                raise ValueError(f"solver.{name} must be positive.")
        if finite_number(self.negative_tolerance, "solver.negative_tolerance") < 0:
            raise ValueError("solver.negative_tolerance must be nonnegative.")
        if isinstance(self.atol, (tuple, list)):
            atol: float | tuple[float, ...] = tuple(finite_number(x, "solver.atol") for x in self.atol)
            if len(atol) != len(STATE_NAMES) or any(x <= 0 for x in atol):
                raise ValueError("solver.atol vector must contain seven positive values.")
        else:
            atol = finite_number(self.atol, "solver.atol")
            if atol <= 0:
                raise ValueError("solver.atol must be positive.")
        for name in ("time_unit", "state_normalization", "inflammation_normalization"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be an explicit nonempty description.")
        object.__setattr__(self, "t_span", span)
        object.__setattr__(self, "t_eval", grid)
        object.__setattr__(self, "atol", atol)


def settings_from_config(draft: ConfigDraft) -> tuple[SimulationSettings, NDArray[np.float64]]:
    """Require every simulation field, reporting all unresolved paths together."""
    check_keys(draft.data, {
        "time_unit", "state_normalization", "inflammation_normalization",
        "time_span", "evaluation_times", "solver", "initial_state",
    }, "simulation")
    sections = {
        "time_span": {"start", "end"},
        "solver": {"method", "rtol", "atol", "max_step", "negative_tolerance"},
        "initial_state": set(STATE_NAMES),
    }
    for section, allowed in sections.items():
        raw = draft.data.get(section)
        if raw is not None:
            check_keys(mapping(raw, section), allowed, section)
    data = draft.require_values(*SIMULATION_REQUIRED_PATHS)
    grid = data["evaluation_times"]
    if not isinstance(grid, (tuple, list)):
        raise ValueError("evaluation_times must be a sequence.")
    settings = SimulationSettings(
        t_span=(data["time_span.start"], data["time_span.end"]),
        t_eval=tuple(grid), method=data["solver.method"], rtol=data["solver.rtol"],
        atol=data["solver.atol"], max_step=data["solver.max_step"],
        negative_tolerance=data["solver.negative_tolerance"],
        time_unit=data["time_unit"], state_normalization=data["state_normalization"],
        inflammation_normalization=data["inflammation_normalization"],
    )
    initial = state_vector([
        finite_number(data[f"initial_state.{name}"], f"initial_state.{name}")
        for name in STATE_NAMES
    ])
    return settings, initial


@dataclass(frozen=True)
class SimulationResult:
    """Raw trajectories in (state, time) order and reproducibility metadata."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    state_names: tuple[str, ...]
    success: bool
    message: str
    nfev: int
    njev: int
    nlu: int
    minimum_state: float
    metadata: dict[str, Any]


class SimulationError(RuntimeError):
    """Integration failed or returned invalid values; no validated result exists."""


def simulate(
    parameters: ModelParameters,
    subject: SubjectParameters,
    external_input: ConstantInput | PiecewiseConstantInput,
    intervention: Intervention,
    initial_state: NDArray[np.float64],
    settings: SimulationSettings,
) -> SimulationResult:
    """Integrate only when explicitly called, restarting at input discontinuities.

    Piecewise inputs are held at each interval's right-continuous value. State
    values remain continuous across jumps. No state values are ever clipped.
    """
    if not isinstance(settings, SimulationSettings):
        raise TypeError("Supply validated SimulationSettings.")
    if not isinstance(external_input, (ConstantInput, PiecewiseConstantInput)):
        raise TypeError("The wrapper supports ConstantInput or PiecewiseConstantInput.")
    current = state_vector(initial_state).copy()
    initial = current.copy()
    # Readiness/domain check before any solve_ivp call.
    rhs(settings.t_span[0], current, parameters, subject, external_input, intervention)
    grid = np.asarray(settings.t_eval, dtype=float)
    start, end = settings.t_span
    boundaries = (start, *(t for t in external_input.breakpoints if start < t < end), end)
    trajectory = np.empty((len(STATE_NAMES), len(grid)), dtype=float)
    nfev = njev = nlu = 0
    messages: list[str] = []
    for left, right in zip(boundaries, boundaries[1:]):
        indices = np.flatnonzero((grid >= left) & (grid <= right))
        requested = grid[indices]
        # The endpoint is required to restart even when absent from output grid.
        segment_grid = np.unique(np.append(requested, right))
        forcing = ConstantInput(external_input(left))
        solution = solve_ivp(
            rhs, (left, right), current,
            args=(parameters, subject, forcing, intervention),
            method=settings.method, rtol=settings.rtol, atol=settings.atol,
            max_step=settings.max_step, t_eval=segment_grid,
        )
        if not solution.success:
            raise SimulationError(f"Integration failed on [{left}, {right}]: {solution.message}")
        values = np.asarray(solution.y, dtype=float)
        times = np.asarray(solution.t, dtype=float)
        if values.shape != (len(STATE_NAMES), len(segment_grid)) or not np.array_equal(times, segment_grid):
            raise SimulationError("Solver returned an incomplete or unexpected output grid.")
        if not np.all(np.isfinite(values)) or np.any(values < -settings.negative_tolerance):
            raise SimulationError("Solver returned nonfinite or negative states beyond the explicit tolerance.")
        trajectory[:, indices] = values[:, np.searchsorted(segment_grid, requested)]
        current = values[:, -1].copy()
        nfev += solution.nfev
        njev += solution.njev
        nlu += solution.nlu
        messages.append(str(solution.message))
    return SimulationResult(
        t=grid, y=trajectory, state_names=STATE_NAMES, success=True,
        message="; ".join(messages), nfev=nfev, njev=njev, nlu=nlu,
        minimum_state=float(np.min(trajectory)),
        metadata={
            "model_version": "0.3", "parameters": asdict(parameters),
            "subject": asdict(subject), "intervention": asdict(intervention),
            "input": {"kind": type(external_input).__name__, **asdict(external_input)},
            "settings": asdict(settings), "initial_state": initial.tolist(),
        },
    )


def simulate_from_config(
    baseline: ConfigDraft, simulation: ConfigDraft, intervention: Intervention,
) -> SimulationResult:
    """Validate all configuration before entering the reusable integrator."""
    validate_baseline(baseline)
    missing: list[str] = []
    baseline_paths = (
        *(f"parameters.{field.name}" for field in fields(ModelParameters)),
        "subject.C_i", *input_required_paths(baseline),
    )
    for draft, paths in ((baseline, baseline_paths), (simulation, SIMULATION_REQUIRED_PATHS)):
        try:
            draft.require_values(*paths)
        except IncompleteConfigurationError as error:
            missing.append(str(error))
    if missing:
        raise IncompleteConfigurationError("\n".join(missing))
    parameters = ModelParameters.from_config(baseline)
    subject = SubjectParameters.from_config(baseline)
    external = input_from_config(baseline)
    settings, initial = settings_from_config(simulation)
    return simulate(parameters, subject, external, intervention, initial, settings)
