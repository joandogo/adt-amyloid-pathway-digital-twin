"""Nonnegative exogenous inflammatory inputs, never ODE states."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

from .parameters import ConfigDraft, check_keys, finite_number, mapping


class InflammatoryInput(Protocol):
    """Callable external input with declared discontinuities."""

    @property
    def breakpoints(self) -> tuple[float, ...]: ...

    def __call__(self, t: float) -> float: ...


@dataclass(frozen=True)
class ConstantInput:
    I_ext: float

    def __post_init__(self) -> None:
        if finite_number(self.I_ext, "input.I_ext") < 0:
            raise ValueError("input.I_ext must be nonnegative.")

    @property
    def breakpoints(self) -> tuple[float, ...]:
        return ()

    def __call__(self, t: float) -> float:
        finite_number(t, "time")
        return self.I_ext


@dataclass(frozen=True)
class PiecewiseConstantInput:
    """Right-continuous steps; n breakpoints require n+1 levels.

    The first level applies before the first breakpoint and the last applies
    after the last. At a breakpoint, the new level applies.
    """

    breakpoints: tuple[float, ...]
    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        points = tuple(finite_number(t, "input.breakpoints") for t in self.breakpoints)
        levels = tuple(finite_number(x, "input.levels") for x in self.levels)
        if len(levels) != len(points) + 1:
            raise ValueError("input.levels must have len(breakpoints) + 1 entries.")
        if any(b <= a for a, b in zip(points, points[1:])):
            raise ValueError("input.breakpoints must be strictly increasing.")
        if any(x < 0 for x in levels):
            raise ValueError("input.levels must be nonnegative.")
        object.__setattr__(self, "breakpoints", points)
        object.__setattr__(self, "levels", levels)

    def __call__(self, t: float) -> float:
        return self.levels[bisect_right(self.breakpoints, finite_number(t, "time"))]


def validate_input_values(data: dict[str, object]) -> None:
    """Check supplied input values, retaining unresolved fields."""
    kind = data.get("kind")
    if kind is not None and kind not in ("constant", "piecewise_constant"):
        raise ValueError("input.kind must be constant or piecewise_constant.")
    allowed = {"kind", "I_ext", "breakpoints", "levels"}
    if kind == "constant":
        allowed = {"kind", "I_ext"}
    elif kind == "piecewise_constant":
        allowed = {"kind", "breakpoints", "levels"}
    check_keys(data, allowed, "input")
    if data.get("I_ext") is not None:
        ConstantInput(data["I_ext"])
    for name in ("breakpoints", "levels"):
        values = data.get(name)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"input.{name} must be a sequence.")
        for value in values:
            if value is not None:
                number = finite_number(value, f"input.{name}")
                if name == "levels" and number < 0:
                    raise ValueError("input.levels must be nonnegative.")
        if name == "breakpoints":
            known = [value for value in values if value is not None]
            if any(b <= a for a, b in zip(known, known[1:])):
                raise ValueError("input.breakpoints must be strictly increasing.")
    points, levels = data.get("breakpoints"), data.get("levels")
    if points is not None and levels is not None:
        if len(levels) != len(points) + 1:
            raise ValueError("input.levels must have len(breakpoints) + 1 entries.")


def input_required_paths(draft: ConfigDraft) -> tuple[str, ...]:
    """List required input paths; an unresolved kind must first be supplied."""
    raw = draft.data.get("input")
    data = mapping({} if raw is None else raw, "input")
    validate_input_values(data)
    kind = data.get("kind")
    if kind == "constant":
        return ("input.kind", "input.I_ext")
    if kind == "piecewise_constant":
        return ("input.kind", "input.breakpoints", "input.levels")
    return ("input.kind",)


def input_from_config(draft: ConfigDraft) -> ConstantInput | PiecewiseConstantInput:
    """Build a complete input without supplying missing levels."""
    paths = input_required_paths(draft)
    draft.require_values(*paths)
    data = mapping(draft.data["input"], "input")
    if data["kind"] == "constant":
        return ConstantInput(data["I_ext"])
    return PiecewiseConstantInput(tuple(data["breakpoints"]), tuple(data["levels"]))
