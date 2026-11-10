"""Seven-state amyloidogenic v0.3 equations, without biological defaults."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .inputs import InflammatoryInput
from .interventions import Intervention
from .parameters import ModelParameters, SubjectParameters, finite_number
from .states import STATE_NAMES


def bace_flux(A: float, I: float, p: ModelParameters) -> float:
    """Inflammation-enhanced BACE processing of the APP pool."""
    return p.k_BACE * A * (1.0 + p.alpha_I * I / (p.K_IB + I))


def hill_activation(value: float, half: float, exponent: float) -> float:
    """Hill activation for nonnegative burden, using stable ratios."""
    if value <= half:
        ratio = (value / half) ** exponent
        return ratio / (1.0 + ratio)
    return 1.0 / (1.0 + (half / value) ** exponent)


def soluble_aggregation_fluxes(A40: float, A42: float, p: ModelParameters) -> tuple[float, float]:
    """Aβ-equivalent transfer from soluble monomers to O."""
    return p.k_O40 * A40 ** p.n_O, p.k_O42 * A42 ** p.n_O


def direct_plaque_fluxes(A40: float, A42: float, p: ModelParameters) -> tuple[float, float]:
    """Aβ-equivalent transfer from soluble monomers directly to P."""
    return p.k_P40 * A40 ** p.n_P, p.k_P42 * A42 ** p.n_P


def state_vector(state: ArrayLike) -> NDArray[np.float64]:
    """Require seven finite nonnegative real states; never clip negatives."""
    raw = np.asarray(state)
    if raw.dtype.kind not in "iuf":
        raise ValueError("States must be real numeric values, not strings or booleans.")
    values = np.asarray(raw, dtype=float)
    if values.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected finite state vector in order {STATE_NAMES}.")
    if np.any(values < 0):
        raise ValueError("Negative state encountered; no RHS clipping is permitted.")
    return values


def rhs(
    t: float,
    state: ArrayLike,
    parameters: ModelParameters,
    subject: SubjectParameters,
    external_input: InflammatoryInput,
    intervention: Intervention,
) -> NDArray[np.float64]:
    """Evaluate E01–E07 with complete, validated parameter objects."""
    if not isinstance(parameters, ModelParameters) or not isinstance(subject, SubjectParameters):
        raise TypeError("RHS requires complete ModelParameters and SubjectParameters objects.")
    if not isinstance(intervention, Intervention):
        raise TypeError("RHS requires a validated Intervention object.")
    finite_number(t, "time")
    A, C99, A40, A42, O, P, I = state_vector(state)
    p = parameters
    C_i = subject.C_i
    I_ext = finite_number(external_input(t), "I_ext(t)")
    if I_ext < 0:
        raise ValueError("I_ext(t) must be nonnegative.")
    products = intervention.apply(p)
    v_BACE = bace_flux(A, I, p)
    v_O40, v_O42 = soluble_aggregation_fluxes(A40, A42, p)
    v_P40, v_P42 = direct_plaque_fluxes(A40, A42, p)
    v_gamma = products.k_gamma_u * C99
    derivative = np.array([
        p.k_syn - v_BACE - p.k_Aout * A,
        v_BACE - (products.k_gamma_u + p.k_deg_C99) * C99,
        products.f40_u * v_gamma - v_O40 - v_P40 - p.k_cl40 * C_i * A40,
        products.f42_u * v_gamma - v_O42 - v_P42 - p.k_cl42 * C_i * A42,
        v_O40 + v_O42 - p.k_OP * O - p.k_clO * C_i * O,
        v_P40 + v_P42 + p.k_OP * O - p.k_clP * C_i * P,
        p.k_Ires * I_ext + p.k_IO * hill_activation(O, p.K_O, p.h_O)
        + p.k_IP * hill_activation(P, p.K_P, p.h_P) - p.k_Ires * I,
    ], dtype=float)
    if not np.all(np.isfinite(derivative)):
        raise ValueError("Nonfinite RHS; review parameters, state scales and solver settings.")
    return derivative
