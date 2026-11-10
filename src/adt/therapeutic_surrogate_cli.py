"""Prepare, select and evaluate the approved P15 E1 therapeutic surrogate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .basal_cli import _json
from .parameters import load_config, mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .states import STATE_NAMES
from .therapeutic_surrogate import (
    CHECKPOINTS,
    FAMILY_ORDER,
    FEATURES,
    METRICS,
    PRIMARY_TARGET,
    TARGET_IDS,
    TREATED_ARMS,
    DomainSpec,
    TargetScales,
    arm_features,
    candidate_specs,
    choose_candidate,
    derived_seed,
    feature_vector,
    make_estimator,
    regression_metrics,
    state_support,
    target_id,
    target_scales,
)


DATASET_FIELDS = (
    'subject_id', 'split', 'cell_id', 'arm', 'metric', 'checkpoint',
    'target_id', 'defined', 'target_reason', 'target_value', *FEATURES,
)
CANDIDATE_FIELDS = (
    'target_id', 'config_id', 'family', 'parameters_json', 'status', 'message',
    'n_train', 'n_validation', 'fit_seconds', 'global_nrmse', 'macro_arm_nrmse',
)
SELECTION_FIELDS = (
    'target_id', 'metric', 'checkpoint', 'status', 'train_defined_fraction',
    'validation_defined_fraction', 'selected_config_id', 'selected_family',
    'validation_global_nrmse', 'validation_macro_arm_nrmse',
    'baseline_macro_arm_nrmse', 'best_nonconstant_macro_arm_nrmse',
    'improvement_over_baseline', 'model_sha256',
)
PREDICTION_FIELDS = (
    'dataset', 'subject_id', 'split', 'cell_id', 'arm', 'target_id',
    'metric', 'checkpoint', 'in_domain', 'domain_reason', 'defined',
    'target_reason', 'true_value', 'prediction', 'prediction_label',
)
METRIC_FIELDS = (
    'dataset', 'target_id', 'metric', 'checkpoint', 'status', 'n_total',
    'n_in_domain', 'n_defined', 'domain_coverage', 'global_nrmse',
    'macro_arm_nrmse', 'worst_arm_nrmse', 'p95_over_global_scale',
    'sign_concordance', 'boundary_nrmse', 'finite_predictions',
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict:
    return mapping(json.loads(path.read_text(encoding='utf-8')), str(path))


def _pinned(path: Path, expected: str, label: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    _require(sha256(raw) == expected, f'{label} SHA-256 mismatch.')
    return raw, mapping(json.loads(raw), label)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames or ()), list(reader)


def _csv_bytes(fields: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _save_progress(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class SurrogatePlan:
    profile: str
    raw: bytes
    config_hash: str
    data: dict
    output_root: Path
    jt1_dir: Path
    jt1_summary_raw: bytes
    jt1_summary_hash: str
    jt1_subjects_hash: str
    jt1_initial_hash: str
    jt1_contrasts_hash: str
    p12_dir: Path
    p12_summary_raw: bytes
    p12_summary: dict
    p12_summary_hash: str
    p12_contrasts_hash: str
    pilot_dir: Path
    pilot_summary_raw: bytes
    pilot_inputs_raw: bytes
    pilot_inputs: dict

    @property
    def models(self) -> dict:
        return mapping(self.data['models'], 'models')

    @property
    def acceptance(self) -> dict:
        return mapping(self.data['acceptance'], 'acceptance')


def _load_plan(path: Path) -> SurrogatePlan:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _require(set(data) == {'profile', 'status', 'source', 'state_order', 'dataset',
                           'targets', 'models', 'selection', 'acceptance', 'domain',
                           'external', 'archive', 'output_root'},
             'Unexpected P15 E1 configuration fields.')
    _require(data['profile'] == 'P15_E1' and data['status'] == 'selected'
             and data['state_order'] == list(STATE_NAMES),
             'E1 requires selected P15_E1 and the v0.3 state order.')
    source = mapping(data['source'], 'source')
    expected_source = {
        'jt1_collection', 'jt1_summary_sha256', 'jt1_subjects_sha256',
        'jt1_initial_states_sha256', 'jt1_contrasts_sha256', 'p12_collection',
        'p12_summary_sha256', 'p12_contrasts_sha256', 'p12_pilot_dir',
        'p12_pilot_summary_sha256', 'p12_pilot_inputs_sha256',
    }
    _require(set(source) == expected_source, 'E1 source lineage is incomplete.')
    jt1_dir = Path(source['jt1_collection']).resolve()
    jt1_summary_raw, jt1 = _pinned(
        jt1_dir / 'summary.json', source['jt1_summary_sha256'], 'JT1 collection')
    _require(jt1_dir.is_relative_to(
                Path('results/interventions/joint_final/collection').resolve())
             and jt1.get('profile') == 'P15_JT1_A2FULL'
             and jt1.get('status') == 'confirmed'
             and jt1.get('numerically_confirmed') is True
             and jt1.get('subjects_selected') == 1024
             and jt1.get('batches_confirmed') == 8
             and jt1.get('integrations_archived') == 24792
             and jt1.get('agreement_rows') == jt1.get('agreement_rows_passing') == 16528
             and jt1.get('contrast_rows') == 229376,
             'E1 requires the pinned fully confirmed JT1 collection.')
    _require(sha256((jt1_dir / 'subjects.csv').read_bytes())
                 == source['jt1_subjects_sha256']
             and sha256((jt1_dir / 'initial_states.csv').read_bytes())
                 == source['jt1_initial_states_sha256']
             and sha256((jt1_dir / 'paired_contrasts.csv').read_bytes())
                 == source['jt1_contrasts_sha256'],
             'Pinned JT1 dataset files changed.')

    p12_dir = Path(source['p12_collection']).resolve()
    p12_summary_raw, p12 = _pinned(
        p12_dir / 'summary.json', source['p12_summary_sha256'], 'P12 collection')
    _require(p12_dir.is_relative_to(
                Path('results/interventions/paired/collection').resolve())
             and p12.get('profile') == 'P12_C1_paired'
             and p12.get('status') == 'confirmed'
             and p12.get('subjects_confirmed') == 120
             and p12.get('contrast_rows') == 26880
             and sha256((p12_dir / 'all_paired_contrasts.csv').read_bytes())
                 == source['p12_contrasts_sha256'],
             'E1 requires the pinned confirmed P12 collection.')
    pilot_dir = Path(source['p12_pilot_dir']).resolve()
    pilot_summary_raw, pilot = _pinned(
        pilot_dir / 'summary.json', source['p12_pilot_summary_sha256'], 'P12 pilot source')
    pilot_inputs_raw, pilot_inputs = _pinned(
        pilot_dir / 'inputs.json', source['p12_pilot_inputs_sha256'], 'P12 pilot inputs')
    _require(pilot.get('status') == 'confirmed'
             and pilot.get('numerically_confirmed') is True
             and pilot.get('inputs_sha256') == source['p12_pilot_inputs_sha256']
             and pilot_inputs.get('state_order') == list(STATE_NAMES),
             'P12 pilot lineage changed.')

    dataset = mapping(data['dataset'], 'dataset')
    _require(dataset == {
        'unit': 'subject',
        'splits': {'train': 640, 'validation': 192, 'test': 192},
        'subjects_per_cell': {'train': 10, 'validation': 3, 'test': 3},
        'treated_arms': list(TREATED_ARMS),
        'rows_per_target': {'train': 4480, 'validation': 1344, 'test': 1344},
        'feature_columns': list(FEATURES),
        'forbidden_features': ['subject_id', 'p14_index', 'cell_id', 'batch',
                               'split', 'path', 'timestamp', 'control_value',
                               'refined_outcome', 'surrogate_prediction'],
        'target_field': 'percent_change_from_control',
        'undefined_reason_field': 'contrast_reason',
        'minimum_defined_fraction': .9,
        'primary_requires_complete_train_validation': True,
    }, 'E1 dataset design differs from approved D0-B.')
    targets = mapping(data['targets'], 'targets')
    _require(targets == {
        'strategy': 'independent_metric_checkpoint', 'metrics': list(METRICS),
        'checkpoints': list(CHECKPOINTS), 'count': 32,
        'primary': {'metric': 'AUC_A42', 'checkpoint': 2000.},
        'responder_classification': 'disabled',
    }, 'E1 target design differs from approved D0-B.')
    models = mapping(data['models'], 'models')
    _require(models == {
        'families': list(FAMILY_ORDER), 'mlp_enabled': False,
        'target_scaling': 'standard_train_only',
        'ridge_alpha': [1e-6, 1e-3, 1., 1e3, 1e6],
        'polynomial_degree': 2, 'polynomial_interaction_only': False,
        'hist_gradient_boosting': {
            'learning_rate': [.03, .1], 'max_leaf_nodes': [7, 15, 31],
            'l2_regularization': [0., 1.], 'max_iter': 300,
            'min_samples_leaf': 20, 'early_stopping': False,
        },
        'configurations_per_target': 23, 'total_fit_budget': 736,
    } and len(candidate_specs(models)) == 23,
             'E1 model budget differs from approved D0-B.')
    selection = mapping(data['selection'], 'selection')
    _require(selection == {
        'metric': 'macro_arm_nrmse', 'tie_relative_tolerance': .01,
        'complexity_order': list(FAMILY_ORDER),
        'primary_improvement_over_baseline': .10,
        'refit_with_validation': False,
        'test_open_policy': 'explicit_once_after_frozen_selection',
    }, 'E1 selection policy changed.')
    _require(mapping(data['external'], 'external') == {
        'dataset': 'P12_C1_paired', 'use_for_selection': False,
        'hard_acceptance_gate': False, 'evaluate_after_internal_test': True,
        'report_domain_rejections': True,
    }, 'E1 external-test policy changed.')
    _require(mapping(data['acceptance'], 'acceptance') == {
        'primary': {
            'global_nrmse_max': .2, 'macro_arm_nrmse_max': .25,
            'each_arm_nrmse_max': .35,
            'p95_absolute_error_over_global_scale_max': .5,
            'sign_concordance_min': .95,
            'sign_band_over_global_scale': .05,
            'boundary_nrmse_max': .3,
            'internal_domain_coverage_min': 1.,
        },
        'secondary': {
            'median_global_nrmse_max': .25,
            'targets_required_below_nrmse': 28,
            'target_nrmse_threshold': .4,
            'worst_global_nrmse_max': .6,
        },
        'require_finite_predictions': True,
    }, 'E1 acceptance thresholds changed after D0-B approval.')
    _require(mapping(data['domain'], 'domain') == {
        'factor_ranges': {
            'k_BACE_multiplier': [1.25, 2.], 'C_i': [.5, .75],
            'I_ext': [.25, 1.]},
        'state_support': 'full_jt1_minmax_feature_only',
        'allowed_protocol': {
            'horizon': 2000., 'checkpoints': list(CHECKPOINTS)},
        'reject_incomplete': True, 'clip_outside_domain': False,
    }, 'E1 applicability domain changed after D0-B approval.')
    _require(mapping(data['archive'], 'archive') == {
        'prediction_label': 'surrogate_predicted',
        'model_format': 'python_pickle', 'snapshot_source': True,
        'hash_all_artifacts': True,
    }, 'E1 archive policy changed.')
    output_root = Path(data['output_root']).resolve()
    _require(data['output_root'] == 'results/surrogate/therapeutic'
             and output_root.is_relative_to(Path('results').resolve())
             and output_root != Path('results').resolve(),
             'E1 output must remain under results/surrogate/therapeutic.')
    return SurrogatePlan(
        data['profile'], raw, sha256(raw), data, output_root,
        jt1_dir, jt1_summary_raw, source['jt1_summary_sha256'],
        source['jt1_subjects_sha256'], source['jt1_initial_states_sha256'],
        source['jt1_contrasts_sha256'], p12_dir, p12_summary_raw, p12,
        source['p12_summary_sha256'], source['p12_contrasts_sha256'],
        pilot_dir, pilot_summary_raw, pilot_inputs_raw, pilot_inputs,
    )


def _dataset_rows(plan: SurrogatePlan) -> tuple[dict[str, bytes], dict]:
    initial_fields, initial_rows = _read_csv(plan.jt1_dir / 'initial_states.csv')
    required_initial = {'subject_id', 'cell_id', 'split', 'k_BACE_multiplier',
                        'C_i', 'I_ext', *STATE_NAMES}
    _require(required_initial <= set(initial_fields) and len(initial_rows) == 1024,
             'JT1 initial-state table is incomplete.')
    subjects: dict[str, dict[str, str]] = {}
    split_subjects = {'train': set(), 'validation': set(), 'test': set()}
    for row in initial_rows:
        subject_id = row['subject_id']
        split = row['split']
        _require(subject_id not in subjects and split in split_subjects,
                 'JT1 initial-state subject/split duplication.')
        subjects[subject_id] = row
        split_subjects[split].add(subject_id)
    _require({name: len(items) for name, items in split_subjects.items()}
             == {'train': 640, 'validation': 192, 'test': 192},
             'JT1 subject split counts changed.')
    support = state_support(initial_rows)

    contrast_fields, contrasts = _read_csv(plan.jt1_dir / 'paired_contrasts.csv')
    required_contrasts = {'unit_id', 'split', 'cell_id', 'k_BACE_multiplier',
                          'C_i', 'I_ext', 'arm', 'intervention_kind', 'strength',
                          'checkpoint', 'metric', 'percent_change_from_control',
                          'contrast_reason'}
    _require(required_contrasts <= set(contrast_fields) and len(contrasts) == 229376,
             'JT1 contrast table is incomplete.')
    output = {'train': [], 'validation': [], 'test': []}
    keys = set()
    arm_catalog: dict[str, tuple[float, float, float, float]] = {}
    for raw in contrasts:
        subject_id = raw['unit_id']
        _require(subject_id in subjects, f'Unknown JT1 contrast subject: {subject_id}.')
        source = subjects[subject_id]
        split = source['split']
        _require(raw['split'] == split and raw['cell_id'] == source['cell_id']
                 and all(float(raw[name]) == float(source[name])
                         for name in ('k_BACE_multiplier', 'C_i', 'I_ext')),
                 f'{subject_id}: contrast metadata differs from initial states.')
        arm = raw['arm']
        strength = float(raw['strength'])
        encoding = arm_features(arm, raw['intervention_kind'], strength)
        if arm in arm_catalog:
            _require(arm_catalog[arm] == encoding, f'{arm}: inconsistent encoding.')
        arm_catalog[arm] = encoding
        metric = raw['metric']
        checkpoint = float(raw['checkpoint'])
        identifier = target_id(metric, checkpoint)
        key = (subject_id, arm, identifier)
        _require(key not in keys, f'Duplicate JT1 surrogate row: {key}.')
        keys.add(key)
        target_value = raw['percent_change_from_control']
        reason = raw['contrast_reason']
        defined = target_value != ''
        _require(defined == (reason == ''),
                 f'{subject_id}/{arm}/{identifier}: value/reason mismatch.')
        if split != 'test' and defined:
            _require(isfinite(float(target_value)),
                     f'{subject_id}/{arm}/{identifier}: nonfinite target.')
        record = {
            'subject_id': subject_id, 'split': split, 'cell_id': source['cell_id'],
            'arm': arm, 'metric': metric, 'checkpoint': checkpoint,
            'target_id': identifier, 'defined': defined,
            'target_reason': reason, 'target_value': target_value,
            'k_BACE_multiplier': source['k_BACE_multiplier'],
            'C_i': source['C_i'], 'I_ext': source['I_ext'],
            **{name: source[name] for name in STATE_NAMES},
            'is_GSI': encoding[0], 'rho_GSI': encoding[1],
            'is_GSM': encoding[2], 'delta_f42_GSM': encoding[3],
        }
        output[split].append(record)
    expected = {'train': 143360, 'validation': 43008, 'test': 43008}
    _require({name: len(rows) for name, rows in output.items()} == expected
             and len(keys) == 229376 and set(arm_catalog) == set(TREATED_ARMS),
             'E1 dataset row coverage changed.')
    files = {f'{name}.csv': _csv_bytes(DATASET_FIELDS, rows)
             for name, rows in output.items()}
    domain = {
        'factor_ranges': mapping(plan.data['domain'], 'domain')['factor_ranges'],
        'state_ranges': {name: list(bounds) for name, bounds in support.items()},
        'arm_encodings': {name: list(arm_catalog[name]) for name in TREATED_ARMS},
        'horizon': 2000., 'checkpoints': list(CHECKPOINTS),
    }
    return files, domain


def _prepare_dataset(plan: SurrogatePlan, check_only: bool) -> int:
    files, domain = _dataset_rows(plan)
    if check_only:
        print('P15 E1 dataset preflight OK: 1024 sujetos; '
              '143360 train, 43008 validation, 43008 test filas.')
        print('Test enrutado y sellado sin calcular estadísticas de sus targets.')
        print('Sin ajuste, predicción ni archivos nuevos.')
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = plan.output_root / 'dataset' / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / 'therapeutic_surrogate.yaml').write_bytes(plan.raw)
    (output / 'jt1_summary.json').write_bytes(plan.jt1_summary_raw)
    (output / 'p12_summary.json').write_bytes(plan.p12_summary_raw)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    _json(output / 'domain.json', domain)
    sources = _source_files()
    for name, raw in sources.items():
        copy = output / 'implementation' / 'adt' / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(raw)
    summary = {
        'profile': plan.profile, 'stage': 'dataset', 'status': 'prepared',
        'test_targets_sealed': True, 'test_target_statistics_computed': False,
        'subjects': {'train': 640, 'validation': 192, 'test': 192},
        'rows': {'train': 143360, 'validation': 43008, 'test': 43008},
        'targets': len(TARGET_IDS), 'arms': len(TREATED_ARMS),
        'config_sha256': plan.config_hash,
        'jt1_summary_sha256': plan.jt1_summary_hash,
        'jt1_subjects_sha256': plan.jt1_subjects_hash,
        'jt1_initial_states_sha256': plan.jt1_initial_hash,
        'jt1_contrasts_sha256': plan.jt1_contrasts_hash,
        'p12_summary_sha256': plan.p12_summary_hash,
        'files_sha256': {name: sha256(raw) for name, raw in files.items()},
        'domain_sha256': sha256((output / 'domain.json').read_bytes()),
        'source_sha256': {name: sha256(raw) for name, raw in sources.items()},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _save_progress(output / 'summary.json', summary)
    print('prepared: 1024 sujetos; 229376 filas; test sellado')
    print(f'Artifacts: {output}')
    return 0


def _dataset_manifest(plan: SurrogatePlan, directory: Path) -> tuple[dict, str]:
    directory = directory.resolve()
    _require(directory.is_relative_to((plan.output_root / 'dataset').resolve()),
             'E1 dataset directory must be under the configured dataset root.')
    summary_path = directory / 'summary.json'
    raw = summary_path.read_bytes()
    summary = mapping(json.loads(raw), 'E1 dataset summary')
    _require(summary.get('profile') == plan.profile
             and summary.get('stage') == 'dataset'
             and summary.get('status') == 'prepared'
             and summary.get('test_targets_sealed') is True
             and summary.get('test_target_statistics_computed') is False
             and summary.get('config_sha256') == plan.config_hash
             and summary.get('jt1_summary_sha256') == plan.jt1_summary_hash
             and summary.get('jt1_subjects_sha256') == plan.jt1_subjects_hash
             and summary.get('jt1_initial_states_sha256') == plan.jt1_initial_hash
             and summary.get('jt1_contrasts_sha256') == plan.jt1_contrasts_hash
             and summary.get('rows') == {
                 'train': 143360, 'validation': 43008, 'test': 43008},
             'E1 dataset manifest is not the pinned prepared dataset.')
    hashes = mapping(summary.get('files_sha256'), 'dataset files_sha256')
    _require(set(hashes) == {'train.csv', 'validation.csv', 'test.csv'},
             'E1 dataset file manifest is incomplete.')
    for name, expected in hashes.items():
        _require(sha256((directory / name).read_bytes()) == expected,
                 f'E1 dataset file changed: {name}.')
    _require(sha256((directory / 'domain.json').read_bytes())
                 == summary.get('domain_sha256'),
             'E1 domain manifest changed.')
    return summary, sha256(raw)


def _dataset_split(directory: Path, split: str) -> list[dict[str, str]]:
    fields, rows = _read_csv(directory / f'{split}.csv')
    expected = {'train': 143360, 'validation': 43008, 'test': 43008}
    _require(fields == DATASET_FIELDS and split in expected
             and len(rows) == expected[split],
             f'E1 {split} table has an unexpected schema or row count.')
    seen: set[tuple[str, str, str]] = set()
    subjects: set[str] = set()
    for row in rows:
        _require(row['split'] == split and row['target_id'] in TARGET_IDS
                 and row['arm'] in TREATED_ARMS
                 and target_id(row['metric'], float(row['checkpoint']))
                 == row['target_id'],
                 f'E1 {split} row metadata changed.')
        key = (row['subject_id'], row['arm'], row['target_id'])
        _require(key not in seen, f'Duplicate E1 {split} row: {key}.')
        seen.add(key)
        subjects.add(row['subject_id'])
        defined = row['defined'] == 'True'
        _require(row['defined'] in ('True', 'False')
                 and defined == (row['target_value'] != '')
                 and defined == (row['target_reason'] == ''),
                 f'E1 {split} target definition metadata changed.')
        feature_vector(row)
        if defined:
            _require(isfinite(float(row['target_value'])),
                     f'E1 {split} contains a nonfinite defined target.')
    expected_subjects = {'train': 640, 'validation': 192, 'test': 192}
    _require(len(subjects) == expected_subjects[split]
             and len(seen) == expected[split],
             f'E1 {split} subject coverage changed.')
    return rows


def _target_arrays(rows: list[dict[str, str]], identifier: str,
                   expected_rows: int) -> tuple[np.ndarray, np.ndarray,
                                                 list[str], list[str], float]:
    selected = [row for row in rows if row['target_id'] == identifier]
    _require(len(selected) == expected_rows,
             f'{identifier}: split row count changed.')
    defined = [row for row in selected if row['defined'] == 'True']
    fraction = len(defined) / len(selected)
    x = (np.vstack([feature_vector(row) for row in defined]) if defined
         else np.empty((0, len(FEATURES)), dtype=float))
    y = np.asarray([float(row['target_value']) for row in defined], dtype=float)
    arms = [row['arm'] for row in defined]
    cells = [row['cell_id'] for row in defined]
    _require(x.shape == (len(defined), len(FEATURES))
             and y.shape == (len(defined),),
             f'{identifier}: invalid defined target arrays.')
    return x, y, arms, cells, fraction


def _target_directory_name(identifier: str) -> str:
    _require(identifier in TARGET_IDS and all(character.isalnum() or character == '_'
                                               for character in identifier),
             'Unsafe E1 target identifier.')
    return identifier


def _completed_target(directory: Path, plan: SurrogatePlan,
                      dataset_hash: str, identifier: str) -> dict | None:
    record_path = directory / 'selection.json'
    if not record_path.exists():
        return None
    record = _read_json(record_path)
    _require(record.get('profile') == plan.profile
             and record.get('target_id') == identifier
             and record.get('config_sha256') == plan.config_hash
             and record.get('dataset_summary_sha256') == dataset_hash,
             f'{identifier}: partial selection lineage changed.')
    candidate_path = directory / 'candidates.csv'
    _require(candidate_path.exists()
             and sha256(candidate_path.read_bytes())
                 == record.get('candidates_sha256'),
             f'{identifier}: candidate archive changed.')
    if record.get('status') == 'selected':
        model_path = directory / 'model.pkl'
        _require(model_path.exists()
                 and sha256(model_path.read_bytes()) == record.get('model_sha256'),
                 f'{identifier}: frozen model changed.')
    return record


def _selection_record(identifier: str, plan: SurrogatePlan,
                      dataset_hash: str, train_rows: list[dict[str, str]],
                      validation_rows: list[dict[str, str]],
                      target_directory: Path, progress: ProgressBar) -> dict:
    train_x, train_y, train_arms, _, train_fraction = _target_arrays(
        train_rows, identifier, 4480)
    validation_x, validation_y, validation_arms, validation_cells, validation_fraction = (
        _target_arrays(validation_rows, identifier, 1344))
    minimum = float(plan.data['dataset']['minimum_defined_fraction'])
    sufficient = train_fraction >= minimum and validation_fraction >= minimum
    if identifier == PRIMARY_TARGET:
        sufficient = sufficient and train_fraction == 1. and validation_fraction == 1.
    metric_name, checkpoint_text = identifier.rsplit('__t', 1)
    base = {
        'profile': plan.profile, 'stage': 'selection_target',
        'target_id': identifier, 'metric': metric_name,
        'checkpoint': float(checkpoint_text),
        'train_defined_fraction': train_fraction,
        'validation_defined_fraction': validation_fraction,
        'config_sha256': plan.config_hash,
        'dataset_summary_sha256': dataset_hash,
    }
    if not sufficient:
        candidate_raw = _csv_bytes(CANDIDATE_FIELDS, [])
        (target_directory / 'candidates.csv').write_bytes(candidate_raw)
        record = {
            **base, 'status': 'insufficient_defined_target',
            'message': 'Undefined targets are neither zero-filled nor imputed.',
            'candidates_sha256': sha256(candidate_raw),
            'model_sha256': None,
        }
        _save_progress(target_directory / 'selection.json', record)
        for spec in candidate_specs(plan.models):
            progress.advance(f'{identifier}/{spec.config_id}: omitted')
        return record

    scales = target_scales(train_y, train_arms)
    candidate_rows: list[dict] = []
    fitted: dict[str, object] = {}
    for spec in candidate_specs(plan.models):
        started = perf_counter()
        row = {
            'target_id': identifier, 'config_id': spec.config_id,
            'family': spec.family,
            'parameters_json': json.dumps(spec.parameters, sort_keys=True,
                                          separators=(',', ':')),
            'status': 'failed', 'message': '', 'n_train': len(train_y),
            'n_validation': len(validation_y), 'fit_seconds': '',
            'global_nrmse': '', 'macro_arm_nrmse': '',
        }
        try:
            estimator = make_estimator(
                spec, derived_seed(f'P15_E1_{identifier}_{spec.config_id}',
                                   plan.jt1_summary_hash))
            estimator.fit(train_x, train_y)
            prediction = np.asarray(estimator.predict(validation_x),
                                    dtype=float).reshape(-1)
            metrics = regression_metrics(
                validation_y, prediction, validation_arms, validation_cells,
                scales, float(plan.acceptance['primary'][
                    'sign_band_over_global_scale']))
            _require(metrics['finite_predictions'] is True,
                     'Candidate produced nonfinite validation predictions.')
            row.update({
                'status': 'completed', 'message': '',
                'global_nrmse': metrics['global_nrmse'],
                'macro_arm_nrmse': metrics['macro_arm_nrmse'],
            })
            fitted[spec.config_id] = estimator
        except Exception as error:  # Archive a failed configuration; do not widen the grid.
            row['message'] = f'{type(error).__name__}: {error}'
        row['fit_seconds'] = perf_counter() - started
        candidate_rows.append(row)
        progress.advance(f'{identifier}/{spec.config_id}: {row["status"]}')

    chosen = choose_candidate(candidate_rows,
                              float(plan.data['selection']['tie_relative_tolerance']))
    completed_nonconstant = [row for row in candidate_rows
                             if row['status'] == 'completed'
                             and row['family'] != 'arm_median']
    _require(completed_nonconstant,
             f'{identifier}: every nonconstant candidate failed.')
    baseline = next(row for row in candidate_rows
                    if row['config_id'] == 'arm_median'
                    and row['status'] == 'completed')
    best_nonconstant = min(completed_nonconstant,
                           key=lambda row: float(row['macro_arm_nrmse']))
    baseline_score = float(baseline['macro_arm_nrmse'])
    best_score = float(best_nonconstant['macro_arm_nrmse'])
    improvement = ((baseline_score - best_score) / baseline_score
                   if baseline_score > 1e-12 else 0.)
    selected_spec = next(spec for spec in candidate_specs(plan.models)
                         if spec.config_id == chosen['config_id'])
    packet = {
        'profile': plan.profile, 'target_id': identifier,
        'features': FEATURES, 'candidate': asdict(selected_spec),
        'scales': asdict(scales), 'estimator': fitted[selected_spec.config_id],
        'config_sha256': plan.config_hash,
        'dataset_summary_sha256': dataset_hash,
    }
    model_raw = pickle.dumps(packet, protocol=5)
    candidate_raw = _csv_bytes(CANDIDATE_FIELDS, candidate_rows)
    (target_directory / 'model.pkl').write_bytes(model_raw)
    (target_directory / 'candidates.csv').write_bytes(candidate_raw)
    record = {
        **base, 'status': 'selected',
        'selected_config_id': chosen['config_id'],
        'selected_family': chosen['family'],
        'selected_parameters': selected_spec.parameters,
        'validation_global_nrmse': float(chosen['global_nrmse']),
        'validation_macro_arm_nrmse': float(chosen['macro_arm_nrmse']),
        'baseline_macro_arm_nrmse': baseline_score,
        'best_nonconstant_macro_arm_nrmse': best_score,
        'improvement_over_baseline': improvement,
        'scales': asdict(scales),
        'model_sha256': sha256(model_raw),
        'model_bytes': len(model_raw),
        'candidates_sha256': sha256(candidate_raw),
        'fits_attempted': len(candidate_rows),
        'fits_completed': sum(row['status'] == 'completed'
                              for row in candidate_rows),
    }
    _save_progress(target_directory / 'selection.json', record)
    return record


def _selection_output(plan: SurrogatePlan, dataset_directory: Path,
                      dataset_summary: dict, dataset_hash: str,
                      resume_directory: Path | None) -> tuple[Path, bool]:
    if resume_directory is not None:
        output = resume_directory.resolve()
        _require(output.is_relative_to((plan.output_root / 'selection').resolve()),
                 'Selection resume directory is outside the configured root.')
        run = _read_json(output / 'run.json')
        _require(run.get('profile') == plan.profile
                 and run.get('config_sha256') == plan.config_hash
                 and run.get('dataset_summary_sha256') == dataset_hash
                 and Path(run.get('dataset_directory', '')).resolve()
                     == dataset_directory.resolve(),
                 'Selection resume lineage changed.')
        _require(mapping(run.get('source_sha256'), 'selection source hashes') == {
            name: sha256(raw) for name, raw in _source_files().items()},
            'Selection resume requires the exact archived E1 source.')
        return output, True
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = plan.output_root / 'selection' / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / 'therapeutic_surrogate.yaml').write_bytes(plan.raw)
    _save_progress(output / 'dataset_summary.json', dataset_summary)
    (output / 'domain.json').write_bytes((dataset_directory / 'domain.json').read_bytes())
    sources = _source_files()
    for name, raw in sources.items():
        copy = output / 'implementation' / 'adt' / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(raw)
    _save_progress(output / 'run.json', {
        'profile': plan.profile, 'stage': 'selection', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_directory': _relative(dataset_directory),
        'dataset_summary_sha256': dataset_hash,
        'config_sha256': plan.config_hash,
        'test_opened': False,
        'source_sha256': {name: sha256(raw) for name, raw in sources.items()},
    })
    return output, False


def _select_models(plan: SurrogatePlan, dataset_directory: Path,
                   resume_directory: Path | None, check_only: bool,
                   progress_interval: float) -> int:
    dataset_directory = dataset_directory.resolve()
    dataset_summary, dataset_hash = _dataset_manifest(plan, dataset_directory)
    train_rows = _dataset_split(dataset_directory, 'train')
    validation_rows = _dataset_split(dataset_directory, 'validation')
    if check_only:
        resume_message = ''
        if resume_directory is not None:
            output, _ = _selection_output(
                plan, dataset_directory, dataset_summary, dataset_hash,
                resume_directory)
            completed = sum(
                _completed_target(
                    output / 'targets' / identifier, plan, dataset_hash,
                    identifier) is not None
                for identifier in TARGET_IDS
                if (output / 'targets' / identifier).exists())
            resume_message = f' Reanudación compatible: {completed}/32 targets completos.'
        print('P15 E1 selection preflight OK: 640 train, 192 validation; '
              '32 targets independientes, 23 configuraciones por target.')
        print('736 ajustes máximos; test verificado solo por hash y no abierto.'
              + resume_message)
        print('Sin ajuste, predicción ni archivos nuevos.')
        return 0
    output, resumed = _selection_output(
        plan, dataset_directory, dataset_summary, dataset_hash, resume_directory)
    existing_summary = output / 'summary.json'
    if existing_summary.exists():
        summary = _read_json(existing_summary)
        _require(summary.get('config_sha256') == plan.config_hash
                 and summary.get('dataset_summary_sha256') == dataset_hash,
                 'Completed selection lineage changed.')
        print(f'{summary["status"]}: selección ya cerrada; no se repitieron ajustes')
        print(f'Artifacts: {output}')
        return 0

    progress = ProgressBar(736, label='P15 E1 selección', unit='ajustes',
                           interval=progress_interval)
    progress.start()
    selections: list[dict] = []
    try:
        for identifier in TARGET_IDS:
            target_directory = output / 'targets' / _target_directory_name(identifier)
            if target_directory.exists():
                complete = _completed_target(
                    target_directory, plan, dataset_hash, identifier)
                if complete is not None:
                    selections.append(complete)
                    for spec in candidate_specs(plan.models):
                        progress.advance(f'{identifier}/{spec.config_id}: reutilizado')
                    continue
                archive = output / 'incomplete_targets' / (
                    f'{identifier}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}')
                archive.parent.mkdir(parents=True, exist_ok=True)
                target_directory.replace(archive)
            target_directory.mkdir(parents=True, exist_ok=False)
            selections.append(_selection_record(
                identifier, plan, dataset_hash, train_rows, validation_rows,
                target_directory, progress))
    finally:
        progress.close()

    selection_rows = []
    for record in selections:
        selection_rows.append({
            'target_id': record['target_id'], 'metric': record['metric'],
            'checkpoint': record['checkpoint'], 'status': record['status'],
            'train_defined_fraction': record['train_defined_fraction'],
            'validation_defined_fraction': record['validation_defined_fraction'],
            'selected_config_id': record.get('selected_config_id', ''),
            'selected_family': record.get('selected_family', ''),
            'validation_global_nrmse': record.get('validation_global_nrmse', ''),
            'validation_macro_arm_nrmse': record.get(
                'validation_macro_arm_nrmse', ''),
            'baseline_macro_arm_nrmse': record.get(
                'baseline_macro_arm_nrmse', ''),
            'best_nonconstant_macro_arm_nrmse': record.get(
                'best_nonconstant_macro_arm_nrmse', ''),
            'improvement_over_baseline': record.get(
                'improvement_over_baseline', ''),
            'model_sha256': record.get('model_sha256', ''),
        })
    selection_raw = _csv_bytes(SELECTION_FIELDS, selection_rows)
    (output / 'selections.csv').write_bytes(selection_raw)
    run_record = _read_json(output / 'run.json')
    source_hashes = mapping(run_record.get('source_sha256'),
                            'selection source hashes')
    current_source_hashes = {
        name: sha256(raw) for name, raw in _source_files().items()}
    _require(source_hashes == current_source_hashes,
             'E1 source changed during model selection.')
    primary = next(record for record in selections
                   if record['target_id'] == PRIMARY_TARGET)
    all_selected = all(record['status'] == 'selected' for record in selections)
    improvement_required = float(
        plan.data['selection']['primary_improvement_over_baseline'])
    primary_gate = (primary['status'] == 'selected'
                    and primary['improvement_over_baseline']
                    >= improvement_required)
    allowed = all_selected and primary_gate
    status = 'selected' if allowed else 'validation_gate_not_met'
    summary = {
        'profile': plan.profile, 'stage': 'selection', 'status': status,
        'models_frozen': True, 'test_opened': False,
        'test_open_allowed': allowed,
        'targets_planned': 32, 'targets_selected': sum(
            record['status'] == 'selected' for record in selections),
        'configurations_per_target': 23,
        'fits_attempted': sum(int(record.get('fits_attempted', 0))
                              for record in selections),
        'primary_target': PRIMARY_TARGET,
        'primary_improvement_over_baseline': primary.get(
            'improvement_over_baseline'),
        'primary_improvement_required': improvement_required,
        'primary_validation_gate_passed': primary_gate,
        'config_sha256': plan.config_hash,
        'dataset_directory': _relative(dataset_directory),
        'dataset_summary_sha256': dataset_hash,
        'selections_sha256': sha256(selection_raw),
        'target_selection_sha256': {
            record['target_id']: sha256((
                output / 'targets' / record['target_id'] / 'selection.json').read_bytes())
            for record in selections
        },
        'model_sha256': {
            record['target_id']: record.get('model_sha256')
            for record in selections
        },
        'source_sha256': source_hashes,
        'packages': {'python': platform.python_version(),
                     'numpy': version('numpy'), 'scipy': version('scipy'),
                     'scikit-learn': version('scikit-learn')},
        'resumed': resumed,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _save_progress(output / 'summary.json', summary)
    _save_progress(output / 'progress.json', {
        'status': status, 'targets_complete': len(selections),
        'fits_attempted': summary['fits_attempted'],
    })
    print(f'{status}: {summary["targets_selected"]}/32 targets; '
          f'{summary["fits_attempted"]}/736 ajustes intentados')
    print(f'Artifacts: {output}')
    return 0


def _domain_spec(dataset_directory: Path) -> DomainSpec:
    document = _read_json(dataset_directory / 'domain.json')
    _require(set(document) == {'factor_ranges', 'state_ranges', 'arm_encodings',
                               'horizon', 'checkpoints'}
             and document['horizon'] == 2000.
             and document['checkpoints'] == list(CHECKPOINTS),
             'E1 domain document changed.')
    factors = mapping(document['factor_ranges'], 'domain factor ranges')
    states = mapping(document['state_ranges'], 'domain state ranges')
    arms = mapping(document['arm_encodings'], 'domain arm encodings')
    _require(set(factors) == {'k_BACE_multiplier', 'C_i', 'I_ext'}
             and set(states) == set(STATE_NAMES)
             and set(arms) == set(TREATED_ARMS),
             'E1 domain coverage is incomplete.')
    return DomainSpec(
        {name: (float(bounds[0]), float(bounds[1]))
         for name, bounds in factors.items()},
        {name: (float(bounds[0]), float(bounds[1]))
         for name, bounds in states.items()},
        {name: tuple(float(value) for value in arms[name])
         for name in TREATED_ARMS},
    )


def _selection_manifest(plan: SurrogatePlan, directory: Path) -> tuple[
        dict, str, Path, dict, str, dict[str, dict]]:
    directory = directory.resolve()
    _require(directory.is_relative_to((plan.output_root / 'selection').resolve()),
             'E1 selection directory is outside the configured root.')
    summary_raw = (directory / 'summary.json').read_bytes()
    summary = mapping(json.loads(summary_raw), 'E1 selection summary')
    _require(summary.get('profile') == plan.profile
             and summary.get('stage') == 'selection'
             and summary.get('status') == 'selected'
             and summary.get('models_frozen') is True
             and summary.get('test_opened') is False
             and summary.get('test_open_allowed') is True
             and summary.get('primary_validation_gate_passed') is True
             and summary.get('targets_selected') == 32
             and summary.get('fits_attempted') == 736
             and summary.get('config_sha256') == plan.config_hash,
             'E1 selection has not passed the frozen validation gate.')
    current_packages = {
        'python': platform.python_version(), 'numpy': version('numpy'),
        'scipy': version('scipy'), 'scikit-learn': version('scikit-learn')}
    _require(summary.get('packages') == current_packages,
             'E1 evaluation requires the frozen selection package versions.')
    source_hashes = mapping(summary.get('source_sha256'),
                            'frozen selection source hashes')
    _require(source_hashes == {
        name: sha256(raw) for name, raw in _source_files().items()},
        'E1 source changed after model selection; do not open test with new code.')
    _require(sha256((directory / 'selections.csv').read_bytes())
                 == summary.get('selections_sha256'),
             'E1 selections table changed.')
    dataset_directory = Path(summary['dataset_directory']).resolve()
    dataset_summary, dataset_hash = _dataset_manifest(plan, dataset_directory)
    _require(dataset_hash == summary.get('dataset_summary_sha256'),
             'E1 selection points to a different dataset summary.')
    selection_hash = sha256(summary_raw)
    targets = {}
    target_hashes = mapping(summary.get('target_selection_sha256'),
                            'target selection hashes')
    model_hashes = mapping(summary.get('model_sha256'), 'model hashes')
    _require(set(target_hashes) == set(TARGET_IDS)
             and set(model_hashes) == set(TARGET_IDS),
             'E1 frozen target hash manifest is incomplete.')
    for identifier in TARGET_IDS:
        target_directory = directory / 'targets' / _target_directory_name(identifier)
        record = _completed_target(
            target_directory, plan, dataset_hash, identifier)
        _require(record is not None and record.get('status') == 'selected',
                 f'{identifier}: selected model is missing.')
        _require(sha256((target_directory / 'selection.json').read_bytes())
                     == target_hashes[identifier]
                 and record.get('model_sha256') == model_hashes[identifier],
                 f'{identifier}: frozen target manifest changed.')
        targets[identifier] = record
    return (summary, selection_hash, dataset_directory, dataset_summary,
            dataset_hash, targets)


def _load_models(selection_directory: Path, plan: SurrogatePlan,
                 dataset_hash: str, records: dict[str, dict]) -> dict[str, dict]:
    output = {}
    for identifier in TARGET_IDS:
        path = selection_directory / 'targets' / identifier / 'model.pkl'
        raw = path.read_bytes()
        _require(sha256(raw) == records[identifier]['model_sha256'],
                 f'{identifier}: model hash changed before test opening.')
        packet = pickle.loads(raw)
        _require(isinstance(packet, dict)
                 and packet.get('profile') == plan.profile
                 and packet.get('target_id') == identifier
                 and tuple(packet.get('features', ())) == FEATURES
                 and packet.get('config_sha256') == plan.config_hash
                 and packet.get('dataset_summary_sha256') == dataset_hash
                 and packet.get('estimator') is not None,
                 f'{identifier}: invalid frozen model packet.')
        output[identifier] = packet
    return output


def _opening_directories(plan: SurrogatePlan, selection_hash: str) -> list[Path]:
    root = plan.output_root / 'evaluation'
    if not root.exists():
        return []
    matches = []
    for opening in sorted(root.glob('*/opening.json')):
        try:
            record = _read_json(opening)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if record.get('selection_summary_sha256') == selection_hash:
            matches.append(opening.parent.resolve())
    return matches


def _evaluation_output(plan: SurrogatePlan, selection_directory: Path,
                       selection_summary: dict, selection_hash: str,
                       dataset_directory: Path, dataset_summary: dict,
                       dataset_hash: str, resume_directory: Path | None) -> tuple[Path, bool]:
    test_hash = dataset_summary['files_sha256']['test.csv']
    previous = _opening_directories(plan, selection_hash)
    if resume_directory is not None:
        output = resume_directory.resolve()
        _require(output in previous
                 and output.is_relative_to((plan.output_root / 'evaluation').resolve()),
                 'Evaluation resume directory does not match the frozen opening.')
        opening = _read_json(output / 'opening.json')
        _require(opening.get('profile') == plan.profile
                 and opening.get('selection_summary_sha256') == selection_hash
                 and opening.get('dataset_summary_sha256') == dataset_hash
                 and opening.get('test_sha256') == test_hash,
                 'E1 test-opening lineage changed.')
        return output, True
    _require(not previous,
             'This frozen selection already has a test-opening record; resume that '
             'directory instead of opening test again.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = plan.output_root / 'evaluation' / stamp
    output.mkdir(parents=True, exist_ok=False)
    # This marker is intentionally written before test targets are parsed.
    _save_progress(output / 'opening.json', {
        'profile': plan.profile, 'stage': 'test_opening', 'status': 'opened',
        'opened_utc': datetime.now(timezone.utc).isoformat(),
        'selection_directory': _relative(selection_directory),
        'selection_summary_sha256': selection_hash,
        'dataset_directory': _relative(dataset_directory),
        'dataset_summary_sha256': dataset_hash,
        'test_sha256': test_hash,
        'policy': 'explicit_once_after_frozen_selection',
    })
    _save_progress(output / 'selection_summary.json', selection_summary)
    _save_progress(output / 'dataset_summary.json', dataset_summary)
    (output / 'therapeutic_surrogate.yaml').write_bytes(plan.raw)
    return output, False


def _p12_rows(plan: SurrogatePlan) -> list[dict[str, object]]:
    cases = mapping(plan.pilot_inputs['cases'], 'P12 pilot cases')
    baseline = mapping(cases['baseline'], 'P12 baseline case')
    baseline_parameters = mapping(baseline['parameters'], 'P12 baseline parameters')
    baseline_k_bace = float(baseline_parameters['k_BACE'])
    state_by_case: dict[str, dict] = {}
    batch_by_case: dict[str, int] = {}
    arm_catalog: dict[str, tuple[str, float]] = {}
    batches = mapping(plan.p12_summary['batches'], 'P12 collection batches')
    _require(set(batches) == {str(index) for index in range(1, 7)},
             'P12 collection batch manifest changed.')
    for batch in range(1, 7):
        archived = mapping(batches[str(batch)], f'P12 batch {batch}')
        batch_directory = (plan.p12_dir.parent.parent / f'batch_{batch:02d}'
                           / Path(archived['directory']).name).resolve()
        _require(sha256((batch_directory / 'summary.json').read_bytes())
                     == archived['summary_sha256'],
                 f'P12 batch {batch} summary changed.')
        inputs = _read_json(batch_directory / 'inputs.json')
        _require(inputs.get('profile') == 'P12_C1_paired'
                 and inputs.get('batch') == batch
                 and inputs.get('pilot_summary_sha256')
                     == plan.p12_summary['pilot_summary_sha256']
                 and inputs.get('pilot_inputs_sha256')
                     == plan.p12_summary['pilot_inputs_sha256']
                 and inputs.get('state_order') == list(STATE_NAMES),
                 f'P12 batch {batch} input lineage changed.')
        input_arms = mapping(inputs['arms'], f'P12 batch {batch} arms')
        current_catalog = {
            arm: (input_arms[arm]['kind'], float(input_arms[arm]['strength']))
            for arm in TREATED_ARMS
        }
        if arm_catalog:
            _require(current_catalog == arm_catalog,
                     f'P12 batch {batch} arm definitions changed.')
        arm_catalog = current_catalog
        starting = mapping(inputs['starting_states'],
                           f'P12 batch {batch} starting states')
        case_ids = inputs['case_ids']
        _require(isinstance(case_ids, list) and len(case_ids) == 20
                 and set(case_ids) == set(starting),
                 f'P12 batch {batch} case coverage changed.')
        for case_id in case_ids:
            _require(case_id not in state_by_case and case_id in cases,
                     f'Duplicate or unknown P12 case: {case_id}.')
            entry = mapping(starting[case_id], f'P12 starting state {case_id}')
            state = mapping(entry['state'], f'P12 state {case_id}')
            _require(set(state) == set(STATE_NAMES),
                     f'P12 state order changed for {case_id}.')
            state_by_case[case_id] = state
            batch_by_case[case_id] = batch
    _require(len(state_by_case) == 120 and set(arm_catalog) == set(TREATED_ARMS),
             'P12 external source must contain 120 subjects and seven arms.')

    fields, contrasts = _read_csv(plan.p12_dir / 'all_paired_contrasts.csv')
    required = {'batch', 'case_id', 'phenotype', 'arm', 'kind', 'strength',
                'checkpoint', 'metric', 'percent_change_from_control',
                'contrast_reason'}
    _require(required <= set(fields) and len(contrasts) == 26880,
             'P12 external contrast table changed.')
    output = []
    seen = set()
    for raw in contrasts:
        case_id = raw['case_id']
        arm = raw['arm']
        _require(case_id in state_by_case
                 and int(raw['batch']) == batch_by_case[case_id]
                 and arm in arm_catalog
                 and (raw['kind'], float(raw['strength'])) == arm_catalog[arm],
                 f'P12 external metadata changed for {case_id}/{arm}.')
        case = mapping(cases[case_id], f'P12 case {case_id}')
        parameters = mapping(case['parameters'], f'P12 parameters {case_id}')
        subject = mapping(case['subject'], f'P12 subject {case_id}')
        external = mapping(case['external_input'], f'P12 input {case_id}')
        encoding = arm_features(arm, raw['kind'], float(raw['strength']))
        identifier = target_id(raw['metric'], float(raw['checkpoint']))
        key = (case_id, arm, identifier)
        _require(key not in seen, f'Duplicate P12 external row: {key}.')
        seen.add(key)
        target_value = raw['percent_change_from_control']
        reason = raw['contrast_reason']
        defined = target_value != ''
        _require(defined == (reason == ''),
                 f'P12 target definition changed for {key}.')
        if defined:
            _require(isfinite(float(target_value)),
                     f'P12 external target is nonfinite for {key}.')
        output.append({
            'subject_id': case_id, 'split': 'external',
            'cell_id': raw['phenotype'], 'arm': arm,
            'metric': raw['metric'], 'checkpoint': float(raw['checkpoint']),
            'target_id': identifier, 'defined': defined,
            'target_reason': reason, 'target_value': target_value,
            'k_BACE_multiplier': float(parameters['k_BACE']) / baseline_k_bace,
            'C_i': float(subject['C_i']), 'I_ext': float(external['I_ext']),
            **{name: float(state_by_case[case_id][name]) for name in STATE_NAMES},
            'is_GSI': encoding[0], 'rho_GSI': encoding[1],
            'is_GSM': encoding[2], 'delta_f42_GSM': encoding[3],
        })
    _require(len(output) == len(seen) == 26880,
             'P12 external row coverage changed.')
    return output


def _evaluate_rows(dataset_name: str, rows: list[dict], models: dict[str, dict],
                   domain: DomainSpec, progress: ProgressBar,
                   require_boundary: bool) -> tuple[list[dict], list[dict],
                                                     list[dict], dict[str, int], float]:
    predictions: list[dict] = []
    metric_rows: list[dict] = []
    details: list[dict] = []
    rejections: dict[str, int] = {}
    prediction_seconds = 0.
    for identifier in TARGET_IDS:
        selected = [row for row in rows if row['target_id'] == identifier]
        _require(selected, f'{dataset_name}/{identifier}: no evaluation rows.')
        accepted_indices = []
        accepted_features = []
        reasons: dict[int, str | None] = {}
        for index, row in enumerate(selected):
            accepted, reason = domain.accepts(row, row['arm'])
            reasons[index] = reason
            if accepted:
                accepted_indices.append(index)
                accepted_features.append(feature_vector(row))
            else:
                key = str(reason)
                rejections[key] = rejections.get(key, 0) + 1
        packet = models[identifier]
        estimator = packet['estimator']
        estimated: dict[int, float] = {}
        if accepted_features:
            x = np.vstack(accepted_features)
            started = perf_counter()
            values = np.asarray(estimator.predict(x), dtype=float).reshape(-1)
            prediction_seconds += perf_counter() - started
            _require(values.shape == (len(accepted_indices),),
                     f'{dataset_name}/{identifier}: prediction shape changed.')
            estimated = dict(zip(accepted_indices, values, strict=True))
        defined_indices = [index for index in accepted_indices
                           if selected[index]['defined'] is True
                           or selected[index]['defined'] == 'True']
        finite_predictions = all(isfinite(value) for value in estimated.values())
        metrics = None
        status = 'evaluated'
        if not accepted_indices:
            status = 'no_in_domain_rows'
        elif not defined_indices:
            status = 'no_defined_in_domain_targets'
        elif not finite_predictions:
            status = 'nonfinite_predictions'
        else:
            truth = np.asarray([float(selected[index]['target_value'])
                                for index in defined_indices], dtype=float)
            predicted = np.asarray([estimated[index]
                                    for index in defined_indices], dtype=float)
            arms = [selected[index]['arm'] for index in defined_indices]
            cells = [selected[index]['cell_id'] for index in defined_indices]
            scale_data = mapping(packet['scales'], f'{identifier} scales')
            scales = TargetScales(
                float(scale_data['global_scale']),
                {name: float(value)
                 for name, value in mapping(scale_data['by_arm'], 'arm scales').items()},
            )
            if set(arms) != set(TREATED_ARMS):
                status = 'insufficient_defined_arm_coverage'
            else:
                metrics = regression_metrics(
                    truth, predicted, arms, cells, scales, .05,
                    require_boundary=require_boundary)
        for index, row in enumerate(selected):
            prediction = estimated.get(index)
            predictions.append({
                'dataset': dataset_name, 'subject_id': row['subject_id'],
                'split': row['split'], 'cell_id': row['cell_id'],
                'arm': row['arm'], 'target_id': identifier,
                'metric': row['metric'], 'checkpoint': row['checkpoint'],
                'in_domain': index in estimated,
                'domain_reason': reasons[index] or '',
                'defined': row['defined'], 'target_reason': row['target_reason'],
                'true_value': row['target_value'],
                'prediction': '' if prediction is None else prediction,
                'prediction_label': ('surrogate_predicted'
                                     if prediction is not None else ''),
            })
        detail = {
            'dataset': dataset_name, 'target_id': identifier,
            'status': status, 'n_total': len(selected),
            'n_in_domain': len(accepted_indices),
            'n_defined_in_domain': len(defined_indices),
            'domain_coverage': len(accepted_indices) / len(selected),
            'metrics': metrics,
        }
        details.append(detail)
        by_arm = metrics.get('by_arm', {}) if metrics else {}
        metric_rows.append({
            'dataset': dataset_name, 'target_id': identifier,
            'metric': selected[0]['metric'],
            'checkpoint': selected[0]['checkpoint'], 'status': status,
            'n_total': len(selected), 'n_in_domain': len(accepted_indices),
            'n_defined': len(defined_indices),
            'domain_coverage': detail['domain_coverage'],
            'global_nrmse': metrics.get('global_nrmse', '') if metrics else '',
            'macro_arm_nrmse': metrics.get('macro_arm_nrmse', '') if metrics else '',
            'worst_arm_nrmse': (max(record['nrmse'] for record in by_arm.values())
                                if by_arm else ''),
            'p95_over_global_scale': (
                metrics.get('p95_absolute_error_over_global_scale', '')
                if metrics else ''),
            'sign_concordance': (metrics.get('sign_concordance', '')
                                 if metrics else ''),
            'boundary_nrmse': (metrics.get('boundary_nrmse', '')
                               if metrics else ''),
            'finite_predictions': finite_predictions,
        })
        progress.advance(f'{dataset_name}/{identifier}: {status}')
    return predictions, metric_rows, details, rejections, prediction_seconds


def _acceptance(plan: SurrogatePlan, details: list[dict]) -> dict:
    by_target = {record['target_id']: record for record in details}
    primary_record = by_target[PRIMARY_TARGET]
    primary_metrics = primary_record.get('metrics')
    thresholds = mapping(plan.acceptance['primary'], 'primary acceptance')
    primary_checks = {
        'evaluated': primary_record['status'] == 'evaluated',
        'all_test_targets_defined': (
            primary_record['n_defined_in_domain'] == primary_record['n_total']),
        'global_nrmse': bool(primary_metrics) and (
            primary_metrics['global_nrmse'] <= thresholds['global_nrmse_max']),
        'macro_arm_nrmse': bool(primary_metrics) and (
            primary_metrics['macro_arm_nrmse']
            <= thresholds['macro_arm_nrmse_max']),
        'each_arm_nrmse': bool(primary_metrics) and all(
            record['nrmse'] <= thresholds['each_arm_nrmse_max']
            for record in primary_metrics['by_arm'].values()),
        'p95_absolute_error': bool(primary_metrics) and (
            primary_metrics['p95_absolute_error_over_global_scale']
            <= thresholds['p95_absolute_error_over_global_scale_max']),
        'sign_concordance': bool(primary_metrics)
            and primary_metrics['sign_concordance'] is not None
            and primary_metrics['sign_concordance']
                >= thresholds['sign_concordance_min'],
        'boundary_nrmse': bool(primary_metrics)
            and primary_metrics['boundary_nrmse'] is not None
            and primary_metrics['boundary_nrmse']
                <= thresholds['boundary_nrmse_max'],
        'finite_predictions': bool(primary_metrics)
            and primary_metrics['finite_predictions'] is True,
        'domain_coverage': primary_record['domain_coverage']
            >= thresholds['internal_domain_coverage_min'],
    }
    secondary_records = [by_target[identifier] for identifier in TARGET_IDS
                         if identifier != PRIMARY_TARGET]
    secondary_values = [record['metrics']['global_nrmse']
                        for record in secondary_records
                        if record['status'] == 'evaluated'
                        and record.get('metrics')
                        and record['metrics']['finite_predictions'] is True]
    secondary = mapping(plan.acceptance['secondary'], 'secondary acceptance')
    secondary_checks = {
        'all_31_evaluated': len(secondary_values) == 31,
        'median_global_nrmse': len(secondary_values) == 31 and (
            float(np.median(secondary_values))
            <= secondary['median_global_nrmse_max']),
        'targets_below_nrmse': len(secondary_values) == 31 and sum(
            value <= secondary['target_nrmse_threshold']
            for value in secondary_values)
            >= secondary['targets_required_below_nrmse'],
        'worst_global_nrmse': len(secondary_values) == 31 and (
            max(secondary_values) <= secondary['worst_global_nrmse_max']),
        'finite_predictions': len(secondary_values) == 31,
    }
    primary_passed = all(primary_checks.values())
    secondary_passed = all(secondary_checks.values())
    return {
        'profile': plan.profile, 'primary_target': PRIMARY_TARGET,
        'primary_thresholds': thresholds,
        'primary_checks': primary_checks, 'primary_passed': primary_passed,
        'secondary_thresholds': secondary,
        'secondary_summary': {
            'evaluated_targets': len(secondary_values),
            'median_global_nrmse': (float(np.median(secondary_values))
                                    if secondary_values else None),
            'targets_below_nrmse': sum(
                value <= secondary['target_nrmse_threshold']
                for value in secondary_values),
            'worst_global_nrmse': (max(secondary_values)
                                   if secondary_values else None),
        },
        'secondary_checks': secondary_checks,
        'secondary_passed': secondary_passed,
        'accepted_for_within_domain_screening': primary_passed and secondary_passed,
        'failure_policy': 'archive_negative_result_without_retuning',
    }


def _benchmark(models: dict[str, dict], test_rows: list[dict[str, str]]) -> dict:
    matrices = {}
    for identifier in TARGET_IDS:
        rows = [row for row in test_rows if row['target_id'] == identifier]
        matrices[identifier] = np.vstack([feature_vector(row) for row in rows])
    for identifier in TARGET_IDS:
        estimator = models[identifier]['estimator']
        estimator.predict(matrices[identifier][:1])
        estimator.predict(matrices[identifier])
    individual = []
    for _ in range(10):
        started = perf_counter()
        for identifier in TARGET_IDS:
            models[identifier]['estimator'].predict(matrices[identifier][:1])
        individual.append(perf_counter() - started)
    batches = []
    for _ in range(3):
        started = perf_counter()
        for identifier in TARGET_IDS:
            models[identifier]['estimator'].predict(matrices[identifier])
        batches.append(perf_counter() - started)
    batch_seconds = float(np.median(batches))
    requests = len(matrices[PRIMARY_TARGET])
    return {
        'warmup': 'one individual and one full-batch prediction per target',
        'individual_repeats': 10,
        'individual_seconds_per_32_target_request_median': float(
            np.median(individual)),
        'batch_repeats': 3, 'batch_requests_per_repeat': requests,
        'batch_seconds_for_32_targets_median': batch_seconds,
        'batch_seconds_per_32_target_request': batch_seconds / requests,
        'prediction_label': 'surrogate_predicted',
    }


def _mechanistic_timing(plan: SurrogatePlan) -> dict:
    elapsed = 0.
    integrations = 0
    batches = mapping(json.loads(plan.jt1_summary_raw)['batches'], 'JT1 batches')
    for batch in range(1, 9):
        item = mapping(batches[str(batch)], f'JT1 batch {batch}')
        directory = Path(item['directory']).resolve()
        summary_raw = (directory / 'summary.json').read_bytes()
        _require(sha256(summary_raw) == item['summary_sha256'],
                 f'JT1 batch {batch} changed before timing summary.')
        summary = mapping(json.loads(summary_raw), f'JT1 batch {batch} summary')
        inputs = _read_json(directory / 'inputs.json')
        started = datetime.fromisoformat(inputs['started_utc'])
        finished = datetime.fromisoformat(summary['finished_utc'])
        _require(finished >= started and summary.get('status') == 'confirmed',
                 f'JT1 batch {batch} timing is invalid.')
        elapsed += (finished - started).total_seconds()
        integrations += int(summary['integrations_attempted'])
    _require(integrations == 24792,
             'JT1 timing integration count changed.')
    return {
        'source': 'archived JT1 batch wall timestamps',
        'wall_seconds_sum': elapsed, 'integrations': integrations,
        'wall_seconds_per_archived_integration': elapsed / integrations,
        'caveat': ('Includes refined variants, gate, writing and verification; '
                   'it is an operational comparison, not a pure solver benchmark.'),
    }


def _evaluate_models(plan: SurrogatePlan, selection_directory: Path,
                     resume_directory: Path | None, check_only: bool,
                     open_test: bool, progress_interval: float) -> int:
    selection_directory = selection_directory.resolve()
    (selection_summary, selection_hash, dataset_directory, dataset_summary,
     dataset_hash, records) = _selection_manifest(plan, selection_directory)
    openings = _opening_directories(plan, selection_hash)
    if check_only:
        if resume_directory is not None:
            _evaluation_output(
                plan, selection_directory, selection_summary, selection_hash,
                dataset_directory, dataset_summary, dataset_hash,
                resume_directory)
        state = ('sin apertura previa' if not openings else
                 f'apertura previa detectada en {openings[0]}')
        print('P15 E1 evaluation preflight OK: selección congelada, puerta de '
              'validación aprobada y 32 modelos verificados por hash.')
        print(f'Test verificado solo por hash; {state}.')
        print('Sin abrir targets, predecir ni escribir archivos nuevos.')
        return 0
    _require(open_test,
             'Evaluation requires the explicit --open-test acknowledgement.')
    output, resumed = _evaluation_output(
        plan, selection_directory, selection_summary, selection_hash,
        dataset_directory, dataset_summary, dataset_hash, resume_directory)
    if (output / 'summary.json').exists():
        summary = _read_json(output / 'summary.json')
        _require(summary.get('selection_summary_sha256') == selection_hash,
                 'Completed E1 evaluation lineage changed.')
        print(f'{summary["status"]}: evaluación ya cerrada; test no se abrió de nuevo')
        print(f'Artifacts: {output}')
        return 0
    partial_names = (
        'internal_predictions.csv', 'internal_metrics.csv',
        'internal_metrics.json', 'acceptance.json', 'external_predictions.csv',
        'external_metrics.csv', 'external_metrics.json',
        'external_domain_rejections.json', 'performance.json',
    )
    existing = [output / name for name in partial_names if (output / name).exists()]
    if existing:
        archive = output / 'incomplete_outputs' / datetime.now(timezone.utc).strftime(
            '%Y%m%dT%H%M%S_%fZ')
        archive.mkdir(parents=True, exist_ok=False)
        for path in existing:
            path.replace(archive / path.name)

    test_rows = _dataset_split(dataset_directory, 'test')
    models = _load_models(selection_directory, plan, dataset_hash, records)
    domain = _domain_spec(dataset_directory)
    progress = ProgressBar(64, label='P15 E1 evaluación', unit='targets',
                           interval=progress_interval)
    progress.start()
    try:
        (internal_predictions, internal_metric_rows, internal_details,
         internal_rejections, internal_prediction_seconds) = _evaluate_rows(
            'JT1_internal_test', test_rows, models, domain, progress,
            require_boundary=True)
        acceptance = _acceptance(plan, internal_details)
        # P12 is opened only after the internal evaluation has been computed.
        external_rows = _p12_rows(plan)
        (external_predictions, external_metric_rows, external_details,
         external_rejections, external_prediction_seconds) = _evaluate_rows(
            'P12_external_descriptive', external_rows, models, domain, progress,
            require_boundary=False)
    finally:
        progress.close()

    internal_prediction_raw = _csv_bytes(PREDICTION_FIELDS, internal_predictions)
    internal_metric_raw = _csv_bytes(METRIC_FIELDS, internal_metric_rows)
    external_prediction_raw = _csv_bytes(PREDICTION_FIELDS, external_predictions)
    external_metric_raw = _csv_bytes(METRIC_FIELDS, external_metric_rows)
    (output / 'internal_predictions.csv').write_bytes(internal_prediction_raw)
    (output / 'internal_metrics.csv').write_bytes(internal_metric_raw)
    _json(output / 'internal_metrics.json', internal_details)
    _json(output / 'acceptance.json', acceptance)
    (output / 'external_predictions.csv').write_bytes(external_prediction_raw)
    (output / 'external_metrics.csv').write_bytes(external_metric_raw)
    _json(output / 'external_metrics.json', external_details)
    _json(output / 'external_domain_rejections.json', {
        'dataset': 'P12_external_descriptive',
        'hard_acceptance_gate': False,
        'rows': len(external_rows),
        'in_domain_rows': sum(record['n_in_domain'] for record in external_details),
        'rejections': external_rejections,
        'interpretation': ('External structured applicability check only; '
                           'P12 did not select models or thresholds.'),
    })
    benchmark = (_benchmark(models, test_rows) if not internal_rejections else {
        'status': 'not_run_internal_domain_coverage_below_one',
        'prediction_label': 'surrogate_predicted',
    })
    mechanistic = _mechanistic_timing(plan)
    if 'batch_seconds_per_32_target_request' in benchmark:
        benchmark['operational_speedup_vs_archived_integration_batch'] = (
            mechanistic['wall_seconds_per_archived_integration']
            / benchmark['batch_seconds_per_32_target_request'])
    performance = {
        'surrogate_benchmark': benchmark,
        'mechanistic_archive': mechanistic,
        'evaluation_prediction_seconds': {
            'internal': internal_prediction_seconds,
            'external': external_prediction_seconds,
        },
    }
    _json(output / 'performance.json', performance)
    status = ('accepted' if acceptance['accepted_for_within_domain_screening']
              else 'not_accepted')
    artifacts = {
        'internal_predictions.csv': sha256(internal_prediction_raw),
        'internal_metrics.csv': sha256(internal_metric_raw),
        'internal_metrics.json': sha256((output / 'internal_metrics.json').read_bytes()),
        'acceptance.json': sha256((output / 'acceptance.json').read_bytes()),
        'external_predictions.csv': sha256(external_prediction_raw),
        'external_metrics.csv': sha256(external_metric_raw),
        'external_metrics.json': sha256((output / 'external_metrics.json').read_bytes()),
        'external_domain_rejections.json': sha256(
            (output / 'external_domain_rejections.json').read_bytes()),
        'performance.json': sha256((output / 'performance.json').read_bytes()),
    }
    summary = {
        'profile': plan.profile, 'stage': 'evaluation', 'status': status,
        'test_opened': True, 'test_opened_once': True,
        'retuning_after_test': False,
        'accepted_for_within_domain_screening': acceptance[
            'accepted_for_within_domain_screening'],
        'primary_passed': acceptance['primary_passed'],
        'secondary_passed': acceptance['secondary_passed'],
        'internal_rows': len(test_rows), 'external_rows': len(external_rows),
        'internal_domain_rejections': internal_rejections,
        'external_hard_gate': False,
        'config_sha256': plan.config_hash,
        'dataset_summary_sha256': dataset_hash,
        'selection_summary_sha256': selection_hash,
        'test_sha256': dataset_summary['files_sha256']['test.csv'],
        'artifact_sha256': artifacts, 'resumed_opening': resumed,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _save_progress(output / 'summary.json', summary)
    print(f'{status}: test interno abierto una vez; P12 evaluado descriptivamente')
    print(f'Artifacts: {output}')
    return 0


def _unique_stage(root: Path, stage: str, status: str) -> Path:
    matches = []
    if root.exists():
        for path in sorted(root.glob('*/summary.json')):
            try:
                summary = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if summary.get('stage') == stage and summary.get('status') == status:
                matches.append(path.parent.resolve())
    _require(len(matches) == 1,
             f'Exactly one {stage}/{status} directory is required; '
             'provide its explicit path.')
    return matches[0]


def _positive_interval(value: str) -> float:
    number = float(value)
    if not 0. < number < float('inf'):
        raise argparse.ArgumentTypeError('progress interval must be positive and finite')
    return number


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description='Prepare, select and evaluate the approved P15 E1 surrogate.')
    command.add_argument('--config', type=Path,
                         default=Path('config/therapeutic_surrogate.yaml'))
    actions = command.add_subparsers(dest='action', required=True)

    prepare = actions.add_parser('prepare', help='Freeze the routed E1 dataset.')
    prepare.add_argument('--check-only', action='store_true')

    select = actions.add_parser(
        'select', help='Fit and freeze models using train/validation only.')
    source = select.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-dir', type=Path)
    source.add_argument('--from-root', action='store_true')
    select.add_argument('--resume-dir', type=Path)
    select.add_argument('--check-only', action='store_true')
    select.add_argument('--progress-interval', type=_positive_interval, default=5.)

    evaluate = actions.add_parser(
        'evaluate', help='Open internal test once and then describe P12 externally.')
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument('--selection-dir', type=Path)
    source.add_argument('--from-root', action='store_true')
    evaluate.add_argument('--resume-dir', type=Path)
    evaluate.add_argument('--check-only', action='store_true')
    evaluate.add_argument('--open-test', action='store_true')
    evaluate.add_argument('--progress-interval', type=_positive_interval, default=5.)
    return command


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    try:
        plan = _load_plan(args.config)
        if args.action == 'prepare':
            return _prepare_dataset(plan, args.check_only)
        if args.action == 'select':
            dataset_directory = (_unique_stage(
                plan.output_root / 'dataset', 'dataset', 'prepared')
                if args.from_root else args.dataset_dir)
            return _select_models(
                plan, dataset_directory, args.resume_dir, args.check_only,
                args.progress_interval)
        if args.check_only and args.open_test:
            raise ValueError('--check-only does not accept --open-test.')
        selection_directory = (_unique_stage(
            plan.output_root / 'selection', 'selection', 'selected')
            if args.from_root else args.selection_dir)
        return _evaluate_models(
            plan, selection_directory, args.resume_dir, args.check_only,
            args.open_test, args.progress_interval)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError,
            pickle.UnpicklingError) as error:
        command.error(str(error))
    return 2
