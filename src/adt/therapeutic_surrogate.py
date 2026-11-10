"""Pure data, model and metric helpers for the selected P15 E1 surrogate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURES = (
    'k_BACE_multiplier', 'C_i', 'I_ext',
    'A', 'C99', 'A40', 'A42', 'O', 'P', 'I',
    'is_GSI', 'rho_GSI', 'is_GSM', 'delta_f42_GSM',
)
METRICS = (
    'AUC_A42', 'A42_final', 'Ratio_42_40', 'AUC_O',
    'P_final', 'AUC_I', 'C99_max', 'AUC_C99',
)
CHECKPOINTS = (200., 500., 1000., 2000.)
TREATED_ARMS = (
    'GSI_01', 'GSI_02', 'GSI_03', 'GSI_04',
    'GSM_low', 'GSM_medium', 'GSM_high',
)
FAMILY_ORDER = ('arm_median', 'ridge', 'polynomial_ridge',
                'hist_gradient_boosting')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def derived_seed(domain: str, lineage: str) -> int:
    """Return a deterministic 32-bit estimator seed."""
    digest = hashlib.sha256(f'{domain}|{lineage}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big', signed=False)


def target_id(metric: str, checkpoint: float) -> str:
    _require(metric in METRICS and checkpoint in CHECKPOINTS,
             'Unknown E1 metric/checkpoint target.')
    return f'{metric}__t{int(checkpoint)}'


TARGET_IDS = tuple(target_id(metric, checkpoint)
                   for metric in METRICS for checkpoint in CHECKPOINTS)
PRIMARY_TARGET = target_id('AUC_A42', 2000.)


def arm_features(arm: str, kind: str, strength: float) -> tuple[float, float, float, float]:
    """Encode GSI activity and GSM fraction modulation separately."""
    _require(arm in TREATED_ARMS and kind in ('GSI', 'GSM')
             and isfinite(strength) and 0. < strength < 1.,
             f'Invalid treated arm: {arm}/{kind}/{strength}.')
    if kind == 'GSI':
        _require(arm.startswith('GSI_'), f'{arm}: GSI kind/name mismatch.')
        return 1., float(strength), 0., 0.
    _require(arm.startswith('GSM_'), f'{arm}: GSM kind/name mismatch.')
    return 0., 0., 1., float(strength)


def feature_vector(row: Mapping[str, object]) -> np.ndarray:
    values = np.asarray([float(row[name]) for name in FEATURES], dtype=float)
    _require(values.shape == (len(FEATURES),) and np.all(np.isfinite(values)),
             'E1 feature vector must contain 14 finite values.')
    return values


def arm_key_from_vector(row: Sequence[float]) -> tuple[int, float, int, float]:
    values = np.asarray(row, dtype=float)
    _require(values.shape == (len(FEATURES),), 'Unexpected E1 feature width.')
    key = (int(round(values[-4])), float(values[-3]),
           int(round(values[-2])), float(values[-1]))
    _require((key[0], key[2]) in ((1, 0), (0, 1)),
             'Feature row does not identify exactly one intervention mechanism.')
    return key


class PerArmMedianRegressor:
    """Deterministic training-median baseline keyed by mechanistic arm encoding."""

    def __init__(self) -> None:
        self.medians_: dict[tuple[int, float, int, float], float] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'PerArmMedianRegressor':
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        _require(x_array.ndim == 2 and x_array.shape[1] == len(FEATURES)
                 and y_array.shape == (len(x_array),)
                 and np.all(np.isfinite(x_array)) and np.all(np.isfinite(y_array)),
                 'Invalid baseline training arrays.')
        keys = [arm_key_from_vector(row) for row in x_array]
        self.medians_ = {
            key: float(np.median(y_array[[item == key for item in keys]]))
            for key in sorted(set(keys))
        }
        _require(len(self.medians_) == len(TREATED_ARMS),
                 'Baseline training data must cover all seven treated arms.')
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        _require(bool(self.medians_), 'PerArmMedianRegressor is not fitted.')
        x_array = np.asarray(x, dtype=float)
        _require(x_array.ndim == 2 and x_array.shape[1] == len(FEATURES),
                 'Invalid baseline prediction array.')
        output = []
        for row in x_array:
            key = arm_key_from_vector(row)
            _require(key in self.medians_, 'Prediction contains an unseen arm encoding.')
            output.append(self.medians_[key])
        return np.asarray(output, dtype=float)


@dataclass(frozen=True)
class CandidateSpec:
    config_id: str
    family: str
    parameters: dict[str, float | int | bool]


def candidate_specs(models: Mapping[str, object]) -> tuple[CandidateSpec, ...]:
    """Expand the approved 1+5+5+12 bounded search exactly."""
    ridge = tuple(float(value) for value in models['ridge_alpha'])
    hist = models['hist_gradient_boosting']
    _require(len(ridge) == 5 and isinstance(hist, Mapping),
             'E1 model grid differs from D0-B.')
    output = [CandidateSpec('arm_median', 'arm_median', {})]
    output.extend(CandidateSpec(f'ridge_a{index:02d}', 'ridge', {'alpha': alpha})
                  for index, alpha in enumerate(ridge, start=1))
    output.extend(CandidateSpec(f'poly2_a{index:02d}', 'polynomial_ridge',
                                {'alpha': alpha, 'degree': 2,
                                 'interaction_only': False})
                  for index, alpha in enumerate(ridge, start=1))
    counter = 0
    for learning_rate in hist['learning_rate']:
        for leaves in hist['max_leaf_nodes']:
            for regularization in hist['l2_regularization']:
                counter += 1
                output.append(CandidateSpec(
                    f'hgb_{counter:02d}', 'hist_gradient_boosting',
                    {'learning_rate': float(learning_rate),
                     'max_leaf_nodes': int(leaves),
                     'l2_regularization': float(regularization),
                     'max_iter': int(hist['max_iter']),
                     'min_samples_leaf': int(hist['min_samples_leaf']),
                     'early_stopping': bool(hist['early_stopping'])},
                ))
    _require(len(output) == 23 and counter == 12,
             'E1 search must contain exactly 23 configurations per target.')
    return tuple(output)


def make_estimator(spec: CandidateSpec, seed: int) -> object:
    if spec.family == 'arm_median':
        return PerArmMedianRegressor()
    alpha = spec.parameters.get('alpha')
    if spec.family == 'ridge':
        regressor = Pipeline([
            ('scale', StandardScaler()),
            ('ridge', Ridge(alpha=float(alpha))),
        ])
    elif spec.family == 'polynomial_ridge':
        regressor = Pipeline([
            ('polynomial', PolynomialFeatures(
                degree=int(spec.parameters['degree']),
                interaction_only=bool(spec.parameters['interaction_only']),
                include_bias=False)),
            ('scale', StandardScaler()),
            ('ridge', Ridge(alpha=float(alpha))),
        ])
    elif spec.family == 'hist_gradient_boosting':
        regressor = HistGradientBoostingRegressor(
            learning_rate=float(spec.parameters['learning_rate']),
            max_leaf_nodes=int(spec.parameters['max_leaf_nodes']),
            l2_regularization=float(spec.parameters['l2_regularization']),
            max_iter=int(spec.parameters['max_iter']),
            min_samples_leaf=int(spec.parameters['min_samples_leaf']),
            early_stopping=bool(spec.parameters['early_stopping']),
            loss='squared_error', random_state=int(seed),
        )
    else:
        raise ValueError(f'Unknown E1 family: {spec.family}.')
    return TransformedTargetRegressor(regressor=regressor,
                                      transformer=StandardScaler())


@dataclass(frozen=True)
class TargetScales:
    global_scale: float
    by_arm: dict[str, float]


def target_scales(y: np.ndarray, arms: Sequence[str]) -> TargetScales:
    values = np.asarray(y, dtype=float)
    _require(values.shape == (len(arms),) and np.all(np.isfinite(values)),
             'Invalid target values for scale calculation.')
    global_scale = max(float(np.std(values)), 1e-12)
    by_arm = {}
    for arm in TREATED_ARMS:
        subset = values[np.asarray([item == arm for item in arms], dtype=bool)]
        _require(len(subset) > 0, f'Training target lacks arm {arm}.')
        by_arm[arm] = max(float(np.std(subset)), 0.1 * global_scale, 1e-12)
    return TargetScales(global_scale, by_arm)


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return sqrt(float(np.mean(np.square(prediction - y))))


def macro_arm_nrmse(y: np.ndarray, prediction: np.ndarray,
                    arms: Sequence[str], scales: TargetScales) -> float:
    values = []
    for arm in TREATED_ARMS:
        mask = np.asarray([item == arm for item in arms], dtype=bool)
        _require(np.any(mask), f'Evaluation target lacks arm {arm}.')
        values.append(_rmse(y[mask], prediction[mask]) / scales.by_arm[arm])
    return float(np.mean(values))


def boundary_cell(cell_id: str) -> bool:
    return len(cell_id) == 4 and cell_id.startswith('C') and any(
        value in ('1', '4') for value in cell_id[1:])


def regression_metrics(y: np.ndarray, prediction: np.ndarray,
                       arms: Sequence[str], cells: Sequence[str],
                       scales: TargetScales, sign_band: float,
                       require_boundary: bool = True) -> dict[str, object]:
    truth = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    _require(truth.shape == predicted.shape == (len(arms),)
             and len(cells) == len(truth) and np.all(np.isfinite(truth)),
             'Invalid E1 metric arrays.')
    finite = bool(np.all(np.isfinite(predicted)))
    if not finite:
        return {'finite_predictions': False}
    absolute = np.abs(predicted - truth)
    rmse = _rmse(truth, predicted)
    arm_records = {}
    for arm in TREATED_ARMS:
        mask = np.asarray([item == arm for item in arms], dtype=bool)
        arm_rmse = _rmse(truth[mask], predicted[mask])
        arm_records[arm] = {
            'n': int(np.sum(mask)),
            'mae': float(np.mean(absolute[mask])),
            'rmse': arm_rmse,
            'nrmse': arm_rmse / scales.by_arm[arm],
        }
    boundary = np.asarray([boundary_cell(value) for value in cells], dtype=bool)
    _require(np.any(boundary) or not require_boundary,
             'E1 evaluation lacks boundary cells.')
    band = sign_band * scales.global_scale
    sign_mask = np.abs(truth) >= band
    sign_value = (float(np.mean(np.sign(truth[sign_mask]) == np.sign(predicted[sign_mask])))
                  if np.any(sign_mask) else None)
    correlation = spearmanr(truth, predicted).statistic
    return {
        'finite_predictions': True,
        'n': len(truth),
        'mae': float(np.mean(absolute)),
        'rmse': rmse,
        'global_nrmse': rmse / scales.global_scale,
        'macro_arm_nrmse': float(np.mean([record['nrmse']
                                         for record in arm_records.values()])),
        'p95_absolute_error': float(np.percentile(absolute, 95)),
        'p95_absolute_error_over_global_scale': (
            float(np.percentile(absolute, 95)) / scales.global_scale),
        'maximum_absolute_error': float(np.max(absolute)),
        'spearman': float(correlation) if isfinite(float(correlation)) else None,
        'sign_band': band,
        'sign_evaluable': int(np.sum(sign_mask)),
        'sign_concordance': sign_value,
        'boundary_n': int(np.sum(boundary)),
        'boundary_nrmse': (_rmse(truth[boundary], predicted[boundary])
                           / scales.global_scale if np.any(boundary) else None),
        'by_arm': arm_records,
    }


def choose_candidate(records: Sequence[Mapping[str, object]],
                     tie_tolerance: float) -> Mapping[str, object]:
    passing = [row for row in records if row.get('status') == 'completed']
    _require(passing, 'No completed candidate models are available.')
    best = min(float(row['macro_arm_nrmse']) for row in passing)
    limit = best + max(abs(best) * tie_tolerance, 1e-12)
    eligible = [row for row in passing if float(row['macro_arm_nrmse']) <= limit]
    order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return min(eligible, key=lambda row: (order[str(row['family'])],
                                          str(row['config_id'])))


@dataclass(frozen=True)
class DomainSpec:
    factor_ranges: dict[str, tuple[float, float]]
    state_ranges: dict[str, tuple[float, float]]
    arm_encodings: dict[str, tuple[float, float, float, float]]

    def accepts(self, features: Mapping[str, object], arm: str) -> tuple[bool, str | None]:
        if arm not in self.arm_encodings:
            return False, 'untrained_arm'
        try:
            values = {name: float(features[name]) for name in FEATURES}
        except (KeyError, TypeError, ValueError):
            return False, 'missing_or_invalid_feature'
        if not all(isfinite(value) for value in values.values()):
            return False, 'nonfinite_feature'
        for name, bounds in self.factor_ranges.items():
            if not bounds[0] <= values[name] <= bounds[1]:
                return False, f'factor_outside_domain:{name}'
        for name, bounds in self.state_ranges.items():
            if not bounds[0] <= values[name] <= bounds[1]:
                return False, f'state_outside_support:{name}'
        actual = tuple(values[name] for name in FEATURES[-4:])
        if actual != self.arm_encodings[arm]:
            return False, 'arm_encoding_mismatch'
        return True, None


def state_support(rows: Iterable[Mapping[str, object]]) -> dict[str, tuple[float, float]]:
    values = {name: [] for name in ('A', 'C99', 'A40', 'A42', 'O', 'P', 'I')}
    count = 0
    for row in rows:
        count += 1
        for name in values:
            value = float(row[name])
            _require(isfinite(value), f'Nonfinite state support value: {name}.')
            values[name].append(value)
    _require(count == 1024, 'JT1 state support requires 1024 subjects.')
    return {name: (min(items), max(items)) for name, items in values.items()}
