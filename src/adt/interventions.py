"""Exact v0.3 control, GSI and GSM transformations."""

from dataclasses import dataclass
from typing import Literal

from .parameters import (
    ConfigDraft, IncompleteConfigurationError, ModelParameters,
    check_keys, finite_number, mapping,
)


@dataclass(frozen=True)
class GammaProducts:
    """Effective gamma activity and its product simplex."""

    k_gamma_u: float
    f40_u: float
    f42_u: float
    fother_u: float


@dataclass(frozen=True)
class Intervention:
    """Constant intervention; strength is rho_GSI or delta_f42_GSM."""

    kind: Literal["control", "GSI", "GSM"]
    strength: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("control", "GSI", "GSM"):
            raise ValueError("Intervention kind must be control, GSI or GSM.")
        if self.kind == "control":
            if self.strength is not None and finite_number(self.strength, "strength") != 0:
                raise ValueError("Control strength must be absent or zero.")
        else:
            if self.strength is None:
                raise IncompleteConfigurationError(f"{self.kind}: supply an explicit strength.")
            _validate_strength(self.strength, self.kind)

    def apply(self, parameters: ModelParameters) -> GammaProducts:
        """Transform activity or fractions without modifying basal parameters."""
        gamma, f40, f42 = parameters.k_gamma, parameters.f40, parameters.f42
        if self.kind == "GSI":
            gamma = (1.0 - self.strength) * gamma
        elif self.kind == "GSM":
            f42 = (1.0 - self.strength) * f42
        return GammaProducts(gamma, f40, f42, 1.0 - (f40 + f42))


def _validate_strength(value: object, name: str) -> None:
    if not 0 <= finite_number(value, name) <= 1:
        raise ValueError(f"{name}: strength must lie in [0, 1].")


def validate_interventions(draft: ConfigDraft) -> None:
    """Validate supplied catalog values without requiring unused levels."""
    check_keys(draft.data, {"control", "GSI", "GSM"}, "interventions")
    for kind, raw in draft.data.items():
        if raw is None:
            continue
        data = mapping(raw, kind)
        key = {"control": "rho_GSI", "GSI": "rho_GSI_levels", "GSM": "delta_f42_GSM"}[kind]
        check_keys(data, {key}, kind)
        value = data.get(key)
        if value is None:
            continue
        if kind == "control":
            Intervention("control", value)
        elif kind == "GSI":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("GSI.rho_GSI_levels must be a nonempty sequence.")
            for level in value:
                if level is not None:
                    _validate_strength(level, "GSI.rho_GSI_levels")
        else:
            levels = mapping(value, "GSM.delta_f42_GSM")
            check_keys(levels, {"low", "medium", "high"}, "GSM.delta_f42_GSM")
            for label, level in levels.items():
                if level is not None:
                    _validate_strength(level, f"GSM.delta_f42_GSM.{label}")


def intervention_from_config(
    draft: ConfigDraft,
    kind: Literal["control", "GSI", "GSM"],
    *,
    level: int | str | None = None,
) -> Intervention:
    """Select GSI by zero-based index or GSM by low/medium/high label."""
    validate_interventions(draft)
    if kind == "control":
        if level is not None:
            raise ValueError("Control does not accept a level.")
        return Intervention("control", draft.require_values("control.rho_GSI")["control.rho_GSI"])
    if kind == "GSM":
        if level not in ("low", "medium", "high"):
            raise ValueError("Select a GSM level: low, medium or high.")
        path = f"GSM.delta_f42_GSM.{level}"
        return Intervention("GSM", draft.require_values(path)[path])
    if kind != "GSI":
        raise ValueError("Intervention kind must be control, GSI or GSM.")
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("Select GSI with a nonnegative integer index.")
    data = mapping(draft.data.get("GSI") or {}, "GSI")
    levels = data.get("rho_GSI_levels")
    if levels is None:
        raise IncompleteConfigurationError("GSI.rho_GSI_levels is missing or null.")
    if level >= len(levels):
        raise ValueError("GSI level index is out of range.")
    if levels[level] is None:
        raise IncompleteConfigurationError(f"GSI.rho_GSI_levels[{level}] is null.")
    return Intervention("GSI", levels[level])
