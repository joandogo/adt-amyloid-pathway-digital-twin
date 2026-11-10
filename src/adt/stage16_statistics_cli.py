"""Archive the approved P16 SA1 design-based statistical analysis."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Mapping

from .basal_cli import _json
from .parameters import load_config, mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .stage16_statistics import (
    MECHANISM_ARMS,
    PRIMARY_CHECKPOINT,
    PRIMARY_METRIC,
    TRADEOFF_METRIC,
    descriptive_contrasts,
    heterogeneity_decomposition,
    mechanistic_bootstrap,
    monotonicity_analysis,
    pareto_frontier,
    surrogate_residual_bootstrap,
    surrogate_residuals,
    temporal_persistence,
)
from .therapeutic_surrogate import CHECKPOINTS, METRICS, PRIMARY_TARGET, TREATED_ARMS


EXPECTED_TOP_LEVEL = {
    'profile', 'status', 'source', 'design', 'estimands', 'summaries',
    'bootstrap', 'monotonicity', 'tradeoff', 'heterogeneity', 'temporal',
    'surrogate', 'inference', 'archive',
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _pinned(path: Path, expected: str, label: str) -> bytes:
    _require(path.is_file(), f'{label} is missing: {path}.')
    raw = path.read_bytes()
    _require(sha256(raw) == expected, f'{label} SHA-256 mismatch.')
    return raw


def _json_object(raw: bytes, label: str) -> dict:
    document = json.loads(raw)
    return mapping(document, label)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _csv_bytes(rows: list[dict], preferred: tuple[str, ...] = ()) -> bytes:
    _require(rows, 'Cannot archive an empty table.')
    names = {name for row in rows for name in row}
    fields = tuple(name for name in preferred if name in names) + tuple(
        sorted(names.difference(preferred)))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


@dataclass(frozen=True)
class AnalysisPlan:
    raw: bytes
    config_hash: str
    output_root: Path
    jt1_dir: Path
    dataset_dir: Path
    selection_dir: Path
    evaluation_dir: Path
    publication_dir: Path
    source_raw: Mapping[str, bytes]
    source_paths: Mapping[str, Path]
    jt1_summary: dict
    dataset_summary: dict
    selection_summary: dict
    evaluation_summary: dict
    acceptance: dict
    rejections: dict
    performance: dict
    zero_band: float
    ddof: int
    bootstrap_replicates: int
    bootstrap_interval: tuple[float, float]
    seed_namespace: str
    endpoints: tuple[dict, ...]
    monotonicity_tolerance: float
    dominance_tolerance: float
    temporal_band: float


def _validate_protocol(data: dict) -> None:
    _require(set(data) == EXPECTED_TOP_LEVEL,
             'Unexpected P16 SA1 configuration fields.')
    _require(data['profile'] == 'P16_SA1' and data['status'] == 'selected',
             'Statistical configuration must be the selected P16_SA1 profile.')
    design = mapping(data['design'], 'design')
    _require(
        design == {
            'unit': 'subject', 'subjects': 1024, 'cells': 64,
            'subjects_per_cell': 16,
            'factors': ['k_BACE_multiplier', 'C_i', 'I_ext'],
            'bins_per_factor': 4, 'treated_arms': list(TREATED_ARMS),
            'checkpoints': list(CHECKPOINTS), 'metrics': list(METRICS),
        }, 'The approved balanced JT1 design changed.')
    estimands = mapping(data['estimands'], 'estimands')
    _require(estimands['primary'] == {
        'metric': PRIMARY_METRIC, 'checkpoint': PRIMARY_CHECKPOINT}
        and estimands['tradeoff'] == {
            'metric': TRADEOFF_METRIC, 'checkpoint': PRIMARY_CHECKPOINT}
        and estimands['contrast_field'] == 'percent_change_from_control',
        'The approved primary or trade-off estimand changed.')
    summaries = mapping(data['summaries'], 'summaries')
    _require(summaries == {
        'standard_deviation_ddof': 1,
        'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95]},
        'The approved summary statistics changed.')
    bootstrap = mapping(data['bootstrap'], 'bootstrap')
    _require(bootstrap['replicates'] == 5000
             and bootstrap['stratification'] == 'cell_id'
             and bootstrap['resampling_unit'] == 'subject'
             and bootstrap['interval_quantiles'] == [0.025, 0.975]
             and bootstrap['seed_namespace'] == 'P16_SA1_bootstrap'
             and bootstrap['label'] == '95% design-bootstrap interval'
             and bootstrap['surrogate_endpoint'] == {
                 'target_id': PRIMARY_TARGET,
                 'statistic': 'median_absolute_residual'},
             'The approved design-bootstrap protocol changed.')
    endpoints = bootstrap['mechanistic_endpoints']
    _require(endpoints == [
        {'metric': PRIMARY_METRIC, 'checkpoint': PRIMARY_CHECKPOINT},
        {'metric': TRADEOFF_METRIC, 'checkpoint': PRIMARY_CHECKPOINT}],
        'The mechanistic bootstrap endpoints changed.')
    monotonicity = mapping(data['monotonicity'], 'monotonicity')
    _require(monotonicity['GSI'] == list(MECHANISM_ARMS['GSI'])
             and monotonicity['GSM'] == list(MECHANISM_ARMS['GSM'])
             and monotonicity['AUC_A42_direction'] == 'nonincreasing'
             and monotonicity['C99_max_direction'] == 'nondecreasing',
             'The mechanism-specific monotonicity protocol changed.')
    tradeoff = mapping(data['tradeoff'], 'tradeoff')
    _require(tradeoff['checkpoint'] == PRIMARY_CHECKPOINT
             and tradeoff['objectives'] == {
                 PRIMARY_METRIC: 'minimize', TRADEOFF_METRIC: 'minimize'},
             'The approved Pareto objectives changed.')
    heterogeneity = mapping(data['heterogeneity'], 'heterogeneity')
    _require(heterogeneity['response'] == {
        'metric': PRIMARY_METRIC, 'checkpoint': PRIMARY_CHECKPOINT}
        and heterogeneity['decomposition'] == 'orthogonal_balanced_4x4x4'
        and heterogeneity['main_arms'] == ['GSI_03', 'GSM_high']
        and heterogeneity['hypothesis_tests'] is False,
        'The approved heterogeneity decomposition changed.')
    surrogate = mapping(data['surrogate'], 'surrogate')
    inference = mapping(data['inference'], 'inference')
    _require(surrogate == {
        'primary_target': PRIMARY_TARGET, 'reuse_frozen_metrics_only': True,
        'allow_refit': False, 'p12_accuracy_metrics': False},
        'The frozen-surrogate policy changed.')
    _require(inference == {
        'population_inference': False, 'primary_p_values': False,
        'interval_interpretation':
            'design_sampling_sensitivity_not_population_confidence'},
        'P16 SA1 must not introduce population inference or primary p-values.')


def _load_plan(path: Path) -> AnalysisPlan:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _validate_protocol(data)
    source = mapping(data['source'], 'source')
    expected_source = {
        'jt1_collection', 'jt1_summary_sha256', 'jt1_subjects_sha256',
        'jt1_contrasts_sha256', 'jt1_agreement_sha256', 'dataset_dir',
        'dataset_summary_sha256', 'selection_dir', 'selection_summary_sha256',
        'evaluation_dir', 'evaluation_summary_sha256',
        'internal_predictions_sha256', 'internal_metrics_sha256',
        'internal_metrics_json_sha256', 'acceptance_sha256',
        'external_rejections_sha256', 'performance_sha256', 'publication_dir',
        'publication_summary_sha256',
    }
    _require(set(source) == expected_source,
             'P16 SA1 source lineage is incomplete or contains unknown fields.')
    jt1_dir = Path(source['jt1_collection']).resolve()
    dataset_dir = Path(source['dataset_dir']).resolve()
    selection_dir = Path(source['selection_dir']).resolve()
    evaluation_dir = Path(source['evaluation_dir']).resolve()
    publication_dir = Path(source['publication_dir']).resolve()
    roots = (
        (jt1_dir, Path('results/interventions/joint_final/collection').resolve()),
        (dataset_dir, Path('results/surrogate/therapeutic/dataset').resolve()),
        (selection_dir, Path('results/surrogate/therapeutic/selection').resolve()),
        (evaluation_dir, Path('results/surrogate/therapeutic/evaluation').resolve()),
        (publication_dir, Path('results/publication/stage15').resolve()),
    )
    _require(all(directory.is_relative_to(root) for directory, root in roots),
             'A P16 source is outside its approved results root.')
    source_paths = {
        'jt1_summary': jt1_dir / 'summary.json',
        'jt1_subjects': jt1_dir / 'subjects.csv',
        'jt1_contrasts': jt1_dir / 'paired_contrasts.csv',
        'jt1_agreement': jt1_dir / 'numerical_agreement.csv',
        'dataset_summary': dataset_dir / 'summary.json',
        'selection_summary': selection_dir / 'summary.json',
        'evaluation_summary': evaluation_dir / 'summary.json',
        'internal_predictions': evaluation_dir / 'internal_predictions.csv',
        'internal_metrics': evaluation_dir / 'internal_metrics.csv',
        'internal_metrics_json': evaluation_dir / 'internal_metrics.json',
        'acceptance': evaluation_dir / 'acceptance.json',
        'external_rejections': evaluation_dir / 'external_domain_rejections.json',
        'performance': evaluation_dir / 'performance.json',
        'publication_summary': publication_dir / 'summary.json',
    }
    expected_hashes = {
        'jt1_summary': source['jt1_summary_sha256'],
        'jt1_subjects': source['jt1_subjects_sha256'],
        'jt1_contrasts': source['jt1_contrasts_sha256'],
        'jt1_agreement': source['jt1_agreement_sha256'],
        'dataset_summary': source['dataset_summary_sha256'],
        'selection_summary': source['selection_summary_sha256'],
        'evaluation_summary': source['evaluation_summary_sha256'],
        'internal_predictions': source['internal_predictions_sha256'],
        'internal_metrics': source['internal_metrics_sha256'],
        'internal_metrics_json': source['internal_metrics_json_sha256'],
        'acceptance': source['acceptance_sha256'],
        'external_rejections': source['external_rejections_sha256'],
        'performance': source['performance_sha256'],
        'publication_summary': source['publication_summary_sha256'],
    }
    source_raw = {name: _pinned(source_paths[name], digest, name)
                  for name, digest in expected_hashes.items()}
    jt1 = _json_object(source_raw['jt1_summary'], 'JT1 summary')
    dataset = _json_object(source_raw['dataset_summary'], 'E1 dataset summary')
    selection = _json_object(source_raw['selection_summary'], 'E1 selection summary')
    evaluation = _json_object(source_raw['evaluation_summary'], 'E1 evaluation summary')
    acceptance = _json_object(source_raw['acceptance'], 'E1 acceptance')
    rejections = _json_object(source_raw['external_rejections'], 'P12 rejections')
    performance = _json_object(source_raw['performance'], 'E1 performance')
    publication = _json_object(source_raw['publication_summary'], 'G8 summary')
    _require(jt1.get('profile') == 'P15_JT1_A2FULL'
             and jt1.get('status') == 'confirmed'
             and jt1.get('numerically_confirmed') is True
             and jt1.get('population_inference') is False
             and jt1.get('subjects_selected') == 1024
             and jt1.get('cells') == 64
             and jt1.get('subjects_per_cell') == 16
             and jt1.get('contrast_rows') == 229376
             and jt1.get('agreement_rows') == jt1.get('agreement_rows_passing') == 16528,
             'SA1 requires the confirmed balanced JT1 archive.')
    _require(dataset.get('stage') == 'dataset' and dataset.get('status') == 'prepared'
             and dataset.get('subjects') == {'train': 640, 'validation': 192,
                                               'test': 192}
             and dataset.get('targets') == 32,
             'SA1 requires the frozen E1 subject split.')
    _require(selection.get('stage') == 'selection'
             and selection.get('status') == 'selected'
             and selection.get('models_frozen') is True
             and selection.get('targets_selected') == 32
             and selection.get('fits_attempted') == 736,
             'SA1 requires the frozen 32-target selection.')
    _require(evaluation.get('stage') == 'evaluation'
             and evaluation.get('status') == 'accepted'
             and evaluation.get('test_opened_once') is True
             and evaluation.get('retuning_after_test') is False
             and evaluation.get('internal_rows') == 43008,
             'SA1 requires the accepted one-time E1 evaluation.')
    _require(acceptance.get('accepted_for_within_domain_screening') is True,
             'The frozen E1 acceptance gate did not pass.')
    _require(rejections.get('rows') == 26880
             and rejections.get('in_domain_rows') == 0
             and rejections.get('hard_acceptance_gate') is False,
             'P12 must remain an out-of-domain descriptive check.')
    _require(publication.get('profile') == 'P15_G8'
             and publication.get('status') == 'rendered'
             and publication.get('scientific_results_recomputed') is False
             and publication.get('figures') == 6,
             'SA1 requires the final visually reviewed G8 package.')

    bootstrap = mapping(data['bootstrap'], 'bootstrap')
    archive = mapping(data['archive'], 'archive')
    _require(set(archive) == {'output_root', 'snapshot_source'}
             and archive['snapshot_source'] is True,
             'SA1 must snapshot its implementation.')
    output_root = Path(archive['output_root']).resolve()
    _require(output_root.is_relative_to(
        Path('results/publication/stage16_analysis').resolve()),
        'SA1 outputs must remain under results/publication/stage16_analysis.')
    return AnalysisPlan(
        raw=raw, config_hash=sha256(raw), output_root=output_root,
        jt1_dir=jt1_dir, dataset_dir=dataset_dir, selection_dir=selection_dir,
        evaluation_dir=evaluation_dir, publication_dir=publication_dir,
        source_raw=source_raw, source_paths=source_paths, jt1_summary=jt1,
        dataset_summary=dataset, selection_summary=selection,
        evaluation_summary=evaluation, acceptance=acceptance,
        rejections=rejections, performance=performance,
        zero_band=float(data['estimands']['zero_band_percentage_points']),
        ddof=int(data['summaries']['standard_deviation_ddof']),
        bootstrap_replicates=int(bootstrap['replicates']),
        bootstrap_interval=tuple(float(value) for value in
                                 bootstrap['interval_quantiles']),
        seed_namespace=str(bootstrap['seed_namespace']),
        endpoints=tuple(mapping(item, 'bootstrap endpoint')
                        for item in bootstrap['mechanistic_endpoints']),
        monotonicity_tolerance=float(data['monotonicity']['tolerance']),
        dominance_tolerance=float(data['tradeoff']['dominance_tolerance']),
        temporal_band=float(data['temporal']['direction_band_percentage_points']),
    )


def _load_rows(plan: AnalysisPlan) -> tuple[
        list[dict[str, str]], list[dict[str, str]], list[dict[str, str]],
        list[dict[str, str]], list[dict]]:
    subjects = _read_csv(plan.jt1_dir / 'subjects.csv')
    contrasts = _read_csv(plan.jt1_dir / 'paired_contrasts.csv')
    predictions = _read_csv(plan.evaluation_dir / 'internal_predictions.csv')
    metrics = _read_csv(plan.evaluation_dir / 'internal_metrics.csv')
    details = json.loads(plan.source_raw['internal_metrics_json'])
    _require(len(subjects) == 1024 and len({row['subject_id'] for row in subjects}) == 1024,
             'JT1 subjects are not the approved 1024 unique units.')
    cell_counts: dict[str, int] = {}
    for row in subjects:
        cell_counts[row['cell_id']] = cell_counts.get(row['cell_id'], 0) + 1
    _require(len(cell_counts) == 64 and set(cell_counts.values()) == {16},
             'JT1 no longer has 16 subjects in each of 64 cells.')
    _require(len(contrasts) == 229376,
             'JT1 contrast row count changed.')
    _require(len(predictions) == 43008
             and sum(row['target_id'] == PRIMARY_TARGET for row in predictions) == 1344,
             'E1 prediction row counts changed.')
    _require(len(metrics) == 32 and all(row['status'] == 'evaluated' for row in metrics)
             and isinstance(details, list) and len(details) == 32,
             'E1 metrics must contain 32 complete frozen targets.')
    return subjects, contrasts, predictions, metrics, details


def _design_rows(plan: AnalysisPlan) -> list[dict]:
    values = (
        ('design', 'synthetic_subjects', 1024),
        ('design', 'joint_cells', 64),
        ('design', 'subjects_per_cell', 16),
        ('design', 'treated_arms', 7),
        ('design', 'checkpoints', 4),
        ('design', 'outcomes', 8),
        ('numerical_audit', 'agreement_rows', 16528),
        ('numerical_audit', 'agreement_rows_passing', 16528),
        ('surrogate', 'train_subjects', 640),
        ('surrogate', 'validation_subjects', 192),
        ('surrogate', 'test_subjects', 192),
        ('surrogate', 'independent_targets', 32),
        ('surrogate', 'candidate_fits', 736),
        ('inference', 'population_inference', False),
        ('inference', 'primary_p_values', False),
    )
    return [{'section': section, 'item': item, 'value': value}
            for section, item, value in values]


def _primary_rows(descriptive: list[dict], bootstrap: list[dict]) -> list[dict]:
    descriptions = {
        (row['arm'], row['metric'], float(row['checkpoint'])): row
        for row in descriptive
        if row['metric'] in (PRIMARY_METRIC, TRADEOFF_METRIC)
        and float(row['checkpoint']) == PRIMARY_CHECKPOINT
    }
    output = []
    for interval in bootstrap:
        key = (interval['arm'], interval['metric'], float(interval['checkpoint']))
        row = descriptions[key]
        output.append({
            'arm': row['arm'], 'mechanism': row['mechanism'],
            'metric': row['metric'], 'checkpoint': row['checkpoint'],
            'n_total': row['n_total'], 'n_defined': row['n_defined'],
            'n_undefined': row['n_undefined'], 'median': row['median'],
            'q1': row['q1'], 'q3': row['q3'], 'iqr': row['iqr'],
            'mean': row['mean'], 'standard_deviation': row['standard_deviation'],
            'median_design_interval_lower': interval['median_interval_lower'],
            'median_design_interval_upper': interval['median_interval_upper'],
            'iqr_design_interval_lower': interval['iqr_interval_lower'],
            'iqr_design_interval_upper': interval['iqr_interval_upper'],
            'bootstrap_replicates': interval['replicates'],
            'bootstrap_seed': interval['seed'],
            'interval_label': '95% design-bootstrap interval',
            'population_confidence_interval': False,
        })
    return output


def _surrogate_table(plan: AnalysisPlan, metrics: list[dict[str, str]]) -> list[dict]:
    primary = next(row for row in metrics if row['target_id'] == PRIMARY_TARGET)
    thresholds = mapping(plan.acceptance['primary_thresholds'], 'primary thresholds')
    checks = mapping(plan.acceptance['primary_checks'], 'primary checks')
    rows = []
    specs = (
        ('global_nrmse', thresholds['global_nrmse_max'], primary['global_nrmse'],
         checks['global_nrmse']),
        ('macro_arm_nrmse', thresholds['macro_arm_nrmse_max'],
         primary['macro_arm_nrmse'], checks['macro_arm_nrmse']),
        ('worst_arm_nrmse', thresholds['each_arm_nrmse_max'],
         primary['worst_arm_nrmse'], checks['each_arm_nrmse']),
        ('p95_over_global_scale',
         thresholds['p95_absolute_error_over_global_scale_max'],
         primary['p95_over_global_scale'], checks['p95_absolute_error']),
        ('sign_concordance', thresholds['sign_concordance_min'],
         primary['sign_concordance'], checks['sign_concordance']),
        ('boundary_nrmse', thresholds['boundary_nrmse_max'],
         primary['boundary_nrmse'], checks['boundary_nrmse']),
        ('domain_coverage', thresholds['internal_domain_coverage_min'],
         primary['domain_coverage'], checks['domain_coverage']),
    )
    for item, threshold, observed, passed in specs:
        rows.append({'category': 'primary_fidelity', 'item': item,
                     'threshold': threshold, 'observed': observed, 'passed': passed,
                     'interpretation': 'frozen_internal_test'})
    secondary = mapping(plan.acceptance['secondary_summary'], 'secondary summary')
    for item, observed in secondary.items():
        rows.append({'category': 'secondary_fidelity', 'item': item,
                     'threshold': '', 'observed': observed,
                     'passed': plan.acceptance['secondary_passed'],
                     'interpretation': 'frozen_internal_test'})
    benchmark = mapping(plan.performance['surrogate_benchmark'], 'surrogate benchmark')
    rows.extend((
        {'category': 'applicability', 'item': 'P12_rows', 'threshold': '',
         'observed': plan.rejections['rows'], 'passed': '',
         'interpretation': 'external_descriptive_only'},
        {'category': 'applicability', 'item': 'P12_in_domain_rows', 'threshold': '',
         'observed': plan.rejections['in_domain_rows'], 'passed': '',
         'interpretation': 'no_external_accuracy_metric'},
        {'category': 'performance', 'item': 'batch_seconds_per_32_target_request',
         'threshold': '',
         'observed': benchmark['batch_seconds_per_32_target_request'], 'passed': '',
         'interpretation': 'operational_hardware_dependent'},
        {'category': 'performance', 'item': 'operational_speedup', 'threshold': '',
         'observed': benchmark['operational_speedup_vs_archived_integration_batch'],
         'passed': '', 'interpretation': 'operational_hardware_dependent'},
    ))
    return rows


def _provenance_rows(plan: AnalysisPlan) -> list[dict]:
    return [
        {'source': name, 'path': _relative(plan.source_paths[name]),
         'sha256': sha256(raw)}
        for name, raw in plan.source_raw.items()
    ]


def _analyse(plan: AnalysisPlan, progress_interval: float) -> Path:
    _, contrasts, predictions, metrics, _ = _load_rows(plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = plan.output_root / stamp
    tables_dir = output / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=False)
    (output / 'stage16_statistical_analysis.yaml').write_bytes(plan.raw)
    started = datetime.now(timezone.utc).isoformat()
    _json(output / 'run.json', {
        'profile': 'P16_SA1', 'stage': 'statistical_analysis',
        'status': 'analyzing', 'started_utc': started,
        'config_sha256': plan.config_hash,
        'source_sha256': {name: sha256(raw) for name, raw in plan.source_raw.items()},
        'rhs_evaluations': 0, 'integrations': 0, 'surrogate_refits': 0,
    })
    progress = ProgressBar(7, label='P16 SA1', unit='análisis',
                           interval=progress_interval)
    progress.start()
    try:
        progress.stage('1/7 descriptivos pareados')
        descriptive = descriptive_contrasts(
            contrasts, zero_band=plan.zero_band, ddof=plan.ddof)
        progress.advance('1/7 descriptivos completos')

        progress.stage('2/7 bootstrap estratificado')
        lineage = sha256(plan.source_raw['jt1_summary'])
        mechanistic_intervals = mechanistic_bootstrap(
            contrasts, endpoints=plan.endpoints,
            replicates=plan.bootstrap_replicates,
            interval=plan.bootstrap_interval, namespace=plan.seed_namespace,
            lineage=lineage)
        surrogate_intervals = surrogate_residual_bootstrap(
            predictions, replicates=plan.bootstrap_replicates,
            interval=plan.bootstrap_interval, namespace=plan.seed_namespace,
            lineage=lineage)
        progress.advance('2/7 bootstrap completo')

        progress.stage('3/7 tendencia por intensidad')
        trends, increments = monotonicity_analysis(
            contrasts, tolerance=plan.monotonicity_tolerance)
        progress.advance('3/7 tendencia completa')

        progress.stage('4/7 frontera mecanística')
        pareto = pareto_frontier(contrasts, tolerance=plan.dominance_tolerance)
        progress.advance('4/7 frontera completa')

        progress.stage('5/7 heterogeneidad 4×4×4')
        heterogeneity = heterogeneity_decomposition(contrasts)
        progress.advance('5/7 heterogeneidad completa')

        progress.stage('6/7 persistencia temporal')
        temporal = temporal_persistence(contrasts, band=plan.temporal_band)
        progress.advance('6/7 persistencia completa')

        progress.stage('7/7 residuales y archivo')
        residuals = surrogate_residuals(predictions)
        table_data = {
            'table_1_design_audit.csv': _csv_bytes(
                _design_rows(plan), ('section', 'item', 'value')),
            'table_2_primary_response.csv': _csv_bytes(
                _primary_rows(descriptive, mechanistic_intervals),
                ('arm', 'mechanism', 'metric', 'checkpoint')),
            'table_3_surrogate_performance.csv': _csv_bytes(
                _surrogate_table(plan, metrics),
                ('category', 'item', 'threshold', 'observed', 'passed',
                 'interpretation')),
            'table_s1_all_descriptives.csv': _csv_bytes(
                descriptive, ('arm', 'mechanism', 'metric', 'checkpoint')),
            'table_s2_monotonicity.csv': _csv_bytes(
                trends, ('mechanism', 'metric', 'direction')),
            'table_s2b_intensity_increments.csv': _csv_bytes(
                increments, ('mechanism', 'metric', 'lower_arm', 'upper_arm')),
            'table_s3_pareto.csv': _csv_bytes(
                pareto, ('arm', 'mechanism')),
            'table_s4_heterogeneity.csv': _csv_bytes(
                heterogeneity, ('arm', 'component')),
            'table_s5_temporal.csv': _csv_bytes(
                temporal, ('arm', 'metric')),
            'table_s6_surrogate_residuals.csv': _csv_bytes(
                residuals, ('arm',)),
            'table_s6b_design_bootstrap.csv': _csv_bytes(
                mechanistic_intervals + surrogate_intervals,
                ('source', 'arm', 'metric', 'checkpoint', 'statistic')),
            'table_s7_undefined.csv': _csv_bytes([
                {'arm': row['arm'], 'mechanism': row['mechanism'],
                 'metric': row['metric'], 'checkpoint': row['checkpoint'],
                 'n_total': row['n_total'], 'n_defined': row['n_defined'],
                 'n_undefined': row['n_undefined'],
                 'undefined_reasons_json': row['undefined_reasons_json']}
                for row in descriptive],
                ('arm', 'mechanism', 'metric', 'checkpoint')),
            'table_s8_numerical_audit.csv': plan.source_raw['jt1_agreement'],
            'table_s9_internal_metrics.csv': plan.source_raw['internal_metrics'],
            'table_s10_provenance.csv': _csv_bytes(
                _provenance_rows(plan), ('source', 'path', 'sha256')),
        }
        for name, raw in table_data.items():
            (tables_dir / name).write_bytes(raw)
        progress.advance('7/7 archivo completo')
    finally:
        progress.close()

    sources = _source_files()
    for name, raw in sources.items():
        destination = output / 'implementation' / 'adt' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    script = Path('scripts/analyze_stage16_results.py')
    destination = output / 'implementation' / 'scripts' / script.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(script.read_bytes())
    archived = [path for path in output.rglob('*') if path.is_file()
                and path.name not in {'run.json', 'summary.json'}]
    summary = {
        'profile': 'P16_SA1', 'stage': 'statistical_analysis',
        'status': 'analyzed', 'design': 'balanced_synthetic_4x4x4',
        'population_inference': False, 'primary_p_values': False,
        'interval_label': '95% design-bootstrap interval',
        'interval_is_population_confidence': False,
        'subjects': 1024, 'cells': 64, 'subjects_per_cell': 16,
        'contrast_rows_read': 229376, 'prediction_rows_read': 43008,
        'bootstrap_replicates': plan.bootstrap_replicates,
        'tables': len(table_data), 'rhs_evaluations': 0,
        'integrations': 0, 'surrogate_refits': 0,
        'config_sha256': plan.config_hash,
        'source_sha256': {name: sha256(raw) for name, raw in plan.source_raw.items()},
        'artifact_sha256': {
            path.relative_to(output).as_posix(): sha256(path.read_bytes())
            for path in archived},
        'packages': {'python': platform.python_version(),
                     'numpy': version('numpy'), 'PyYAML': version('PyYAML')},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run the approved P16 SA1 design-based analysis.')
    parser.add_argument('--config', type=Path,
                        default=Path('config/stage16_statistical_analysis.yaml'))
    parser.add_argument('--check-only', action='store_true',
                        help='Verify frozen sources and row counts without statistics or writes.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    args = parser.parse_args(argv)
    _require(args.progress_interval > 0., 'Progress interval must be positive.')
    plan = _load_plan(args.config)
    if args.check_only:
        _load_rows(plan)
        print('P16 SA1 preflight OK: fuentes JT1/E1/G8 fijadas por hash.')
        print('Plan: 1024 sujetos, 229376 contrastes, 43008 predicciones y '
              '5000 réplicas estratificadas.')
        print('Sin estadística, RHS, integración, reajuste ni archivos nuevos.')
        return 0
    output = _analyse(plan, args.progress_interval)
    print('analyzed: SA1 completo; 15 tablas, sin simulación ni reajuste')
    print(f'Artifacts: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
