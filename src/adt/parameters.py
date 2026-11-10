"""Load incomplete YAML drafts without inventing scientific parameters.

Scientific values remain external to the equations.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Iterator
import warnings

import yaml


class IncompleteConfigurationWarning(UserWarning):
    """A draft contains unresolved values and is not ready for general use."""


class IncompleteConfigurationError(ValueError):
    """Required configuration values are missing or unresolved."""


def _null_paths(value: object, prefix: str = "") -> Iterator[str]:
    """Yield paths of explicit nulls in nested mappings and lists."""
    if value is None:
        yield prefix
    elif isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _null_paths(child, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _null_paths(child, f"{prefix}[{index}]")


@dataclass
class ConfigDraft:
    """YAML data awaiting operation-specific and scientific validation."""

    source: Path
    data: dict[str, object]

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        """Return every explicit null path, including nested list entries."""
        return tuple(_null_paths(self.data))

    def require_values(self, *paths: str) -> dict[str, object]:
        """Require dotted mapping paths, including all values below each path.

        Missing keys and nulls raise before returning any requested values.
        List indices are reported but are not accepted as path syntax.
        """
        selected: dict[str, object] = {}
        missing: list[str] = []
        for path in paths:
            if not path or any(not part for part in path.split(".")):
                raise ValueError("Required paths must be nonempty dotted keys.")
            value: object = self.data
            for key in path.split("."):
                if not isinstance(value, dict) or key not in value:
                    missing.append(path)
                    break
                value = value[key]
            else:
                nulls = tuple(_null_paths(value, path))
                if nulls:
                    missing.extend(nulls)
                else:
                    selected[path] = value
        if missing:
            fields = ", ".join(dict.fromkeys(missing))
            raise IncompleteConfigurationError(
                f"{self.source}: required configuration values are missing or null: "
                f"{fields}. Supply explicit values before this operation."
            )
        return selected


def load_config(
    path: str | Path, *, allow_incomplete: bool = True
) -> ConfigDraft:
    """Load a YAML mapping; warn about nulls or reject them in strict mode.

    An empty document is invalid. No defaults or biological validation are
    applied. Strict mode checks explicit nulls, not schema completeness.
    """
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{source}: expected a nonempty YAML mapping.")
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"{source}: top-level configuration keys must be strings.")
    draft = ConfigDraft(source=source, data=data)
    if draft.unresolved_fields:
        fields = ", ".join(draft.unresolved_fields)
        message = (
            f"{source}: incomplete configuration; null fields: {fields}. "
            "No values were substituted. Supply the fields required by the "
            "selected operation before use."
        )
        if not allow_incomplete:
            raise IncompleteConfigurationError(message)
        warnings.warn(message, IncompleteConfigurationWarning, stacklevel=2)
    return draft


def finite_number(value: object, name: str) -> float:
    """Require a finite real scalar, excluding booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a real number, got {value!r}.")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name}: expected a finite number.")
    return number


def mapping(value: object, name: str) -> dict[str, object]:
    """Require a mapping with string keys."""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name}: expected a mapping with string keys.")
    return value


def check_keys(data: dict[str, object], allowed: set[str], name: str) -> None:
    """Reject misspelled/unsupported keys without treating absence as a value."""
    extra = data.keys() - allowed
    if extra:
        raise ValueError(f"{name}: unknown fields: {', '.join(sorted(extra))}.")


@dataclass(frozen=True)
class ModelParameters:
    """Global v0.3 parameters; no biological defaults."""

    k_syn: float
    k_BACE: float
    k_Aout: float
    alpha_I: float
    K_IB: float
    k_gamma: float
    k_deg_C99: float
    f40: float
    f42: float
    k_O40: float
    k_O42: float
    n_O: float
    k_P40: float
    k_P42: float
    n_P: float
    k_cl40: float
    k_cl42: float
    k_OP: float
    k_clO: float
    k_clP: float
    k_IO: float
    k_IP: float
    k_Ires: float
    K_O: float
    K_P: float
    h_O: float
    h_P: float

    def __post_init__(self) -> None:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        validate_parameter_values(values)
        if any(value is None for value in values.values()):
            pending = [name for name, value in values.items() if value is None]
            raise IncompleteConfigurationError(f"Unresolved parameters: {', '.join(pending)}")

    @property
    def fother(self) -> float:
        """Untracked gamma products complete the product simplex."""
        return 1.0 - (self.f40 + self.f42)

    @classmethod
    def from_config(cls, draft: ConfigDraft) -> ModelParameters:
        """Build complete parameters from a validated baseline draft."""
        validate_baseline(draft)
        names = [field.name for field in fields(cls)]
        values = draft.require_values(*(f"parameters.{name}" for name in names))
        return cls(**{name: finite_number(values[f"parameters.{name}"], name) for name in names})


def validate_parameter_values(data: dict[str, object]) -> None:
    """Validate supplied basal values while retaining null placeholders.

    Basal kinetic rates are positive. k_syn and inflammatory coupling strengths
    may be zero. Exponents and half-activation constants must be positive.
    """
    allowed = {field.name for field in fields(ModelParameters)}
    check_keys(data, allowed, "parameters")
    nonnegative = {"k_syn", "alpha_I", "k_IO", "k_IP", "f40", "f42"}
    values: dict[str, float] = {}
    for name, value in data.items():
        if value is None:
            continue
        number = finite_number(value, f"parameters.{name}")
        if number < 0 or (number == 0 and name not in nonnegative):
            raise ValueError(f"parameters.{name}: must be {'nonnegative' if name in nonnegative else 'positive'}.")
        if name in {"f40", "f42"} and number > 1:
            raise ValueError(f"parameters.{name}: must lie in [0, 1].")
        values[name] = number
    if {"f40", "f42"} <= values.keys() and values["f40"] + values["f42"] > 1:
        raise ValueError("parameters: f40 + f42 must not exceed 1.")
    if {"k_O40", "k_O42"} <= values.keys() and values["k_O42"] <= values["k_O40"]:
        raise ValueError("parameters: basal k_O42 must exceed k_O40.")
    for soluble in ("k_cl40", "k_cl42", "k_clO"):
        if {"k_clP", soluble} <= values.keys() and values["k_clP"] >= values[soluble]:
            raise ValueError(f"parameters: k_clP must be lower than {soluble}.")


@dataclass(frozen=True)
class SubjectParameters:
    """Individual clearance capacity, not a dynamic state."""

    C_i: float

    def __post_init__(self) -> None:
        if finite_number(self.C_i, "subject.C_i") <= 0:
            raise ValueError("subject.C_i must be positive.")

    @classmethod
    def from_config(cls, draft: ConfigDraft) -> SubjectParameters:
        validate_baseline(draft)
        return cls(finite_number(draft.require_values("subject.C_i")["subject.C_i"], "subject.C_i"))


def validate_baseline(draft: ConfigDraft) -> None:
    """Validate supplied baseline fields, including incomplete drafts."""
    from .inputs import validate_input_values

    check_keys(draft.data, {"parameters", "subject", "input"}, "baseline")
    parameters = draft.data.get("parameters")
    if parameters is not None:
        validate_parameter_values(mapping(parameters, "parameters"))
    subject = draft.data.get("subject")
    if subject is not None:
        subject_data = mapping(subject, "subject")
        check_keys(subject_data, {"C_i"}, "subject")
        if subject_data.get("C_i") is not None:
            SubjectParameters(subject_data["C_i"])
    external = draft.data.get("input")
    if external is not None:
        validate_input_values(mapping(external, "input"))


def load_baseline(path: str | Path, *, allow_incomplete: bool = True) -> ConfigDraft:
    """Read and validate a basal draft; strict mode also requires all keys."""
    from .inputs import input_required_paths

    draft = load_config(path)
    validate_baseline(draft)
    if not allow_incomplete:
        draft.require_values(
            *(f"parameters.{field.name}" for field in fields(ModelParameters)),
            "subject.C_i", *input_required_paths(draft),
        )
    return draft
