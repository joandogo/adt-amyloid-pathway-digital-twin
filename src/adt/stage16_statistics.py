"""Pure design-based statistics for the approved P16 SA1 analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import isfinite
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .therapeutic_surrogate import CHECKPOINTS, METRICS, PRIMARY_TARGET, TREATED_ARMS


MECHANISM_ARMS = {
    'GSI': TREATED_ARMS[:4],
    'GSM': TREATED_ARMS[4:],
}
PRIMARY_METRIC = 'AUC_A42'
TRADEOFF_METRIC = 'C99_max'
PRIMARY_CHECKPOINT = 2000.


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def derived_seed(namespace: str, lineage: str, label: str) -> int:
    """Derive an order-independent 64-bit analysis seed."""
    raw = f'{namespace}|{lineage}|{label}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big', signed=False)


def summary_statistics(values: Sequence[float], ddof: int = 1) -> dict[str, float | int]:
    """Return transparent distribution summaries for finite values."""
    array = np.asarray(values, dtype=float)
    _require(array.ndim == 1 and len(array) > ddof and np.all(np.isfinite(array)),
             'Summary values must be a finite vector longer than ddof.')
    q05, q25, median, q75, q95 = np.quantile(
        array, [.05, .25, .5, .75, .95])
    return {
        'n': int(len(array)),
        'mean': float(np.mean(array)),
        'standard_deviation': float(np.std(array, ddof=ddof)),
        'median': float(median),
        'q1': float(q25),
        'q3': float(q75),
        'iqr': float(q75 - q25),
        'p05': float(q05),
        'p95': float(q95),
        'minimum': float(np.min(array)),
        'maximum': float(np.max(array)),
    }


def _defined_contrast(row: Mapping[str, str]) -> bool:
    return (row.get('contrast_reason', '') == ''
            and row.get('percent_change_from_control', '') != '')


def descriptive_contrasts(rows: Iterable[Mapping[str, str]], *,
                          zero_band: float, ddof: int = 1) -> list[dict]:
    """Summarize every arm–metric–checkpoint group without imputing undefined rows."""
    _require(isfinite(zero_band) and zero_band >= 0., 'Zero band must be finite.')
    groups: dict[tuple[str, str, float], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row['arm'], row['metric'], float(row['checkpoint']))
        groups[key].append(row)
    output = []
    for (arm, metric, checkpoint), group in groups.items():
        values = [float(row['percent_change_from_control'])
                  for row in group if _defined_contrast(row)]
        _require(values, f'No defined contrasts for {arm}/{metric}/{checkpoint}.')
        reasons = Counter(row.get('contrast_reason', '') or 'defined' for row in group)
        stats = summary_statistics(values, ddof)
        array = np.asarray(values, dtype=float)
        output.append({
            'arm': arm,
            'mechanism': 'GSI' if arm.startswith('GSI_') else 'GSM',
            'metric': metric,
            'checkpoint': checkpoint,
            'n_total': len(group),
            'n_defined': len(values),
            'n_undefined': len(group) - len(values),
            'undefined_reasons_json': json.dumps(
                {key: value for key, value in sorted(reasons.items())
                 if key != 'defined'}, separators=(',', ':')),
            **stats,
            'negative_fraction': float(np.mean(array < -zero_band)),
            'near_zero_fraction': float(np.mean(np.abs(array) <= zero_band)),
            'positive_fraction': float(np.mean(array > zero_band)),
            'zero_band_percentage_points': zero_band,
        })
    arm_rank = {arm: index for index, arm in enumerate(TREATED_ARMS)}
    metric_rank = {metric: index for index, metric in enumerate(METRICS)}
    checkpoint_rank = {value: index for index, value in enumerate(CHECKPOINTS)}
    return sorted(output, key=lambda row: (
        arm_rank[row['arm']], metric_rank[row['metric']],
        checkpoint_rank[row['checkpoint']]))


def stratified_design_bootstrap(
        rows: Sequence[Mapping[str, str]], *, value: Callable[[Mapping[str, str]], float],
        subject_field: str, replicates: int, seed: int,
        interval: tuple[float, float]) -> dict[str, float | int]:
    """Bootstrap subjects within balanced cells and summarize median and IQR."""
    _require(replicates >= 2, 'Bootstrap requires at least two replicates.')
    lower, upper = interval
    _require(0. <= lower < upper <= 1., 'Invalid bootstrap interval quantiles.')
    cells: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        _require(isfinite(number), 'Bootstrap values must be finite.')
        cells[row['cell_id']].append((row[subject_field], number))
    _require(cells, 'Bootstrap requires at least one cell.')
    ordered_cells = []
    sizes = set()
    for cell_id in sorted(cells):
        records = sorted(cells[cell_id])
        subjects = [subject for subject, _ in records]
        _require(len(subjects) == len(set(subjects)),
                 f'{cell_id}: duplicate bootstrap subject.')
        ordered_cells.append([number for _, number in records])
        sizes.add(len(records))
    _require(len(sizes) == 1 and next(iter(sizes)) > 0,
             'Bootstrap cells must contain the same positive subject count.')
    matrix = np.asarray(ordered_cells, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[1],
                           size=(replicates, matrix.shape[0], matrix.shape[1]))
    sampled = np.take_along_axis(
        np.broadcast_to(matrix, (replicates, *matrix.shape)), indices, axis=2)
    sampled = sampled.reshape(replicates, -1)
    medians = np.median(sampled, axis=1)
    quartiles = np.quantile(sampled, [.25, .75], axis=1)
    iqrs = quartiles[1] - quartiles[0]
    observed = summary_statistics(matrix.ravel())
    return {
        'replicates': replicates,
        'cells': matrix.shape[0],
        'subjects_per_cell': matrix.shape[1],
        'observed_median': observed['median'],
        'median_interval_lower': float(np.quantile(medians, lower)),
        'median_interval_upper': float(np.quantile(medians, upper)),
        'observed_iqr': observed['iqr'],
        'iqr_interval_lower': float(np.quantile(iqrs, lower)),
        'iqr_interval_upper': float(np.quantile(iqrs, upper)),
        'seed': seed,
    }


def mechanistic_bootstrap(
        rows: Sequence[Mapping[str, str]], *, endpoints: Sequence[Mapping[str, object]],
        replicates: int, interval: tuple[float, float], namespace: str,
        lineage: str) -> list[dict]:
    """Return cell-stratified intervals for approved mechanistic endpoints."""
    output = []
    for endpoint in endpoints:
        metric, checkpoint = str(endpoint['metric']), float(endpoint['checkpoint'])
        for arm in TREATED_ARMS:
            selected = [row for row in rows if row['arm'] == arm
                        and row['metric'] == metric
                        and float(row['checkpoint']) == checkpoint
                        and _defined_contrast(row)]
            seed = derived_seed(namespace, lineage, f'mechanistic|{metric}|{checkpoint}|{arm}')
            result = stratified_design_bootstrap(
                selected, value=lambda row: float(row['percent_change_from_control']),
                subject_field='unit_id', replicates=replicates, seed=seed,
                interval=interval)
            output.append({'source': 'mechanistic', 'arm': arm, 'metric': metric,
                           'checkpoint': checkpoint, 'statistic': 'median_and_iqr',
                           **result})
    return output


def monotonicity_analysis(rows: Sequence[Mapping[str, str]], *,
                          tolerance: float) -> tuple[list[dict], list[dict]]:
    """Quantify within-subject intensity trends separately for GSI and GSM."""
    _require(tolerance >= 0. and isfinite(tolerance), 'Invalid monotonicity tolerance.')
    selected = [row for row in rows if row['metric'] in (PRIMARY_METRIC, TRADEOFF_METRIC)
                and float(row['checkpoint']) == PRIMARY_CHECKPOINT
                and _defined_contrast(row)]
    values = {(row['metric'], row['unit_id'], row['arm']):
              float(row['percent_change_from_control']) for row in selected}
    subjects = sorted({row['unit_id'] for row in selected})
    trend_rows, increment_rows = [], []
    for mechanism, arms in MECHANISM_ARMS.items():
        for metric, direction in ((PRIMARY_METRIC, 'nonincreasing'),
                                  (TRADEOFF_METRIC, 'nondecreasing')):
            evaluable, monotone = 0, 0
            increments: dict[tuple[str, str], list[float]] = {
                pair: [] for pair in zip(arms[:-1], arms[1:], strict=True)}
            for subject in subjects:
                keys = [(metric, subject, arm) for arm in arms]
                if not all(key in values for key in keys):
                    continue
                series = np.asarray([values[key] for key in keys], dtype=float)
                differences = np.diff(series)
                evaluable += 1
                if direction == 'nonincreasing':
                    monotone += int(np.all(differences <= tolerance))
                else:
                    monotone += int(np.all(differences >= -tolerance))
                for pair, difference in zip(increments, differences, strict=True):
                    increments[pair].append(float(difference))
            _require(evaluable > 0, f'No complete subjects for {mechanism}/{metric}.')
            trend_rows.append({
                'mechanism': mechanism, 'metric': metric, 'direction': direction,
                'subjects_evaluable': evaluable, 'subjects_monotone': monotone,
                'monotone_fraction': monotone / evaluable, 'tolerance': tolerance,
            })
            for (lower_arm, upper_arm), changes in increments.items():
                increment_rows.append({
                    'mechanism': mechanism, 'metric': metric,
                    'lower_arm': lower_arm, 'upper_arm': upper_arm,
                    **summary_statistics(changes),
                })
    return trend_rows, increment_rows


def pareto_frontier(rows: Sequence[Mapping[str, str]], *, tolerance: float) -> list[dict]:
    """Count subject-level non-dominated arms for Aβ42 reduction and C99 burden."""
    selected = [row for row in rows if row['metric'] in (PRIMARY_METRIC, TRADEOFF_METRIC)
                and float(row['checkpoint']) == PRIMARY_CHECKPOINT
                and _defined_contrast(row)]
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for row in selected:
        values[row['unit_id']][row['arm']][row['metric']] = float(
            row['percent_change_from_control'])
    counts = Counter()
    sizes = Counter()
    evaluable = 0
    for arms in values.values():
        if not all(arm in arms and set(arms[arm]) == {PRIMARY_METRIC, TRADEOFF_METRIC}
                   for arm in TREATED_ARMS):
            continue
        evaluable += 1
        frontier = []
        for arm in TREATED_ARMS:
            a_auc, a_c99 = arms[arm][PRIMARY_METRIC], arms[arm][TRADEOFF_METRIC]
            dominated = False
            for other in TREATED_ARMS:
                if other == arm:
                    continue
                b_auc, b_c99 = arms[other][PRIMARY_METRIC], arms[other][TRADEOFF_METRIC]
                weak = b_auc <= a_auc + tolerance and b_c99 <= a_c99 + tolerance
                strict = b_auc < a_auc - tolerance or b_c99 < a_c99 - tolerance
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                frontier.append(arm)
                counts[arm] += 1
        sizes[len(frontier)] += 1
    _require(evaluable > 0, 'No complete subjects for the Pareto analysis.')
    size_json = json.dumps(dict(sorted(sizes.items())), separators=(',', ':'))
    return [{
        'arm': arm,
        'mechanism': 'GSI' if arm.startswith('GSI_') else 'GSM',
        'subjects_evaluable': evaluable,
        'frontier_count': counts[arm],
        'frontier_fraction': counts[arm] / evaluable,
        'frontier_size_distribution_json': size_json,
        'dominance_tolerance': tolerance,
    } for arm in TREATED_ARMS]


def _cell_indices(cell_id: str) -> tuple[int, int, int]:
    _require(len(cell_id) == 4 and cell_id.startswith('C')
             and all(character in '1234' for character in cell_id[1:]),
             f'Invalid 4×4×4 cell identifier: {cell_id}.')
    return (int(cell_id[1]) - 1, int(cell_id[2]) - 1, int(cell_id[3]) - 1)


def heterogeneity_decomposition(rows: Sequence[Mapping[str, str]]) -> list[dict]:
    """Orthogonally decompose balanced 4×4×4 primary-response variation."""
    selected = [row for row in rows if row['metric'] == PRIMARY_METRIC
                and float(row['checkpoint']) == PRIMARY_CHECKPOINT
                and _defined_contrast(row)]
    output = []
    for arm in TREATED_ARMS:
        cells: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        for row in selected:
            if row['arm'] == arm:
                cells[_cell_indices(row['cell_id'])].append(
                    float(row['percent_change_from_control']))
        _require(len(cells) == 64, f'{arm}: heterogeneity requires all 64 cells.')
        sizes = {len(values) for values in cells.values()}
        _require(len(sizes) == 1 and next(iter(sizes)) > 0,
                 f'{arm}: cell sizes are not balanced.')
        repetitions = next(iter(sizes))
        array = np.empty((4, 4, 4, repetitions), dtype=float)
        for index, values in cells.items():
            array[index] = np.asarray(values, dtype=float)
        grand = float(np.mean(array))
        effect_a = np.mean(array, axis=(1, 2, 3)) - grand
        effect_b = np.mean(array, axis=(0, 2, 3)) - grand
        effect_c = np.mean(array, axis=(0, 1, 3)) - grand
        mean_ab = np.mean(array, axis=(2, 3))
        mean_ac = np.mean(array, axis=(1, 3))
        mean_bc = np.mean(array, axis=(0, 3))
        effect_ab = mean_ab - grand - effect_a[:, None] - effect_b[None, :]
        effect_ac = mean_ac - grand - effect_a[:, None] - effect_c[None, :]
        effect_bc = mean_bc - grand - effect_b[:, None] - effect_c[None, :]
        cell_mean = np.mean(array, axis=3)
        effect_abc = (cell_mean - grand - effect_a[:, None, None]
                      - effect_b[None, :, None] - effect_c[None, None, :]
                      - effect_ab[:, :, None] - effect_ac[:, None, :]
                      - effect_bc[None, :, :])
        sums = {
            'k_BACE': 4 * 4 * repetitions * float(np.sum(effect_a ** 2)),
            'C_i': 4 * 4 * repetitions * float(np.sum(effect_b ** 2)),
            'I_ext': 4 * 4 * repetitions * float(np.sum(effect_c ** 2)),
            'k_BACE:C_i': 4 * repetitions * float(np.sum(effect_ab ** 2)),
            'k_BACE:I_ext': 4 * repetitions * float(np.sum(effect_ac ** 2)),
            'C_i:I_ext': 4 * repetitions * float(np.sum(effect_bc ** 2)),
            'k_BACE:C_i:I_ext': repetitions * float(np.sum(effect_abc ** 2)),
            'within_cell': float(np.sum((array - cell_mean[:, :, :, None]) ** 2)),
        }
        total = float(np.sum((array - grand) ** 2))
        between = sum(value for key, value in sums.items() if key != 'within_cell')
        _require(total > 0. and np.isclose(sum(sums.values()), total,
                                          rtol=1e-9, atol=1e-12),
                 f'{arm}: orthogonal sums do not reconstruct total variation.')
        for component, sum_of_squares in sums.items():
            output.append({
                'arm': arm, 'component': component,
                'sum_of_squares': sum_of_squares,
                'total_sum_of_squares': total,
                'share_of_total': sum_of_squares / total,
                'share_of_between_cell': (
                    sum_of_squares / between if component != 'within_cell' else ''),
                'cells': 64, 'subjects_per_cell': repetitions,
                'hypothesis_test': False,
            })
    return output


def _sign(value: float, band: float) -> int:
    if value < -band:
        return -1
    if value > band:
        return 1
    return 0


def temporal_persistence(rows: Sequence[Mapping[str, str]], *, band: float) -> list[dict]:
    """Summarize checkpoint medians and within-subject direction persistence."""
    defined = [row for row in rows if _defined_contrast(row)]
    output = []
    for arm in TREATED_ARMS:
        for metric in METRICS:
            grouped = {
                checkpoint: [float(row['percent_change_from_control']) for row in defined
                             if row['arm'] == arm and row['metric'] == metric
                             and float(row['checkpoint']) == checkpoint]
                for checkpoint in CHECKPOINTS
            }
            _require(all(grouped.values()), f'Missing temporal values for {arm}/{metric}.')
            medians = [float(np.median(grouped[checkpoint]))
                       for checkpoint in CHECKPOINTS]
            subject_values: dict[str, dict[float, float]] = defaultdict(dict)
            for row in defined:
                if row['arm'] == arm and row['metric'] == metric:
                    subject_values[row['unit_id']][float(row['checkpoint'])] = float(
                        row['percent_change_from_control'])
            complete = [values for values in subject_values.values()
                        if set(values) == set(CHECKPOINTS)]
            _require(complete, f'No complete temporal subjects for {arm}/{metric}.')
            preserved = sum(len({_sign(values[checkpoint], band)
                                 for checkpoint in CHECKPOINTS}) == 1
                            for values in complete)
            output.append({
                'arm': arm, 'metric': metric,
                **{f'median_t{int(checkpoint)}': median
                   for checkpoint, median in zip(CHECKPOINTS, medians, strict=True)},
                'median_change_t200_to_t2000': medians[-1] - medians[0],
                'maximum_absolute_consecutive_median_change': float(
                    np.max(np.abs(np.diff(medians)))),
                'subjects_complete': len(complete),
                'direction_preserved_count': preserved,
                'direction_preserved_fraction': preserved / len(complete),
                'direction_band_percentage_points': band,
            })
    return output


def surrogate_residuals(predictions: Sequence[Mapping[str, str]]) -> list[dict]:
    """Summarize frozen primary-test residuals by arm."""
    selected = [row for row in predictions
                if row['target_id'] == PRIMARY_TARGET
                and row['in_domain'] == 'True' and row['defined'] == 'True'
                and row['prediction'] != '' and row['true_value'] != '']
    output = []
    for arm in TREATED_ARMS:
        residuals = np.asarray([
            float(row['prediction']) - float(row['true_value'])
            for row in selected if row['arm'] == arm], dtype=float)
        stats = summary_statistics(residuals)
        absolute = np.abs(residuals)
        output.append({
            'arm': arm, **stats,
            'median_absolute_residual': float(np.median(absolute)),
            'p95_absolute_residual': float(np.quantile(absolute, .95)),
            'maximum_absolute_residual': float(np.max(absolute)),
        })
    return output


def surrogate_residual_bootstrap(
        predictions: Sequence[Mapping[str, str]], *, replicates: int,
        interval: tuple[float, float], namespace: str, lineage: str) -> list[dict]:
    """Bootstrap the primary median absolute residual within test cells."""
    selected = [row for row in predictions
                if row['target_id'] == PRIMARY_TARGET
                and row['in_domain'] == 'True' and row['defined'] == 'True'
                and row['prediction'] != '' and row['true_value'] != '']
    output = []
    for arm in TREATED_ARMS:
        arm_rows = [row for row in selected if row['arm'] == arm]
        seed = derived_seed(namespace, lineage, f'surrogate|{PRIMARY_TARGET}|{arm}')
        result = stratified_design_bootstrap(
            arm_rows,
            value=lambda row: abs(float(row['prediction']) - float(row['true_value'])),
            subject_field='subject_id', replicates=replicates, seed=seed,
            interval=interval)
        output.append({
            'source': 'surrogate', 'arm': arm, 'metric': PRIMARY_METRIC,
            'checkpoint': PRIMARY_CHECKPOINT,
            'statistic': 'median_absolute_residual', **result,
        })
    return output
