"""Collect eight confirmed P15 JT1-A2FULL batches without integrating."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median

import yaml

from .basal_cli import _json
from .final_cohort_cli import _candidate_case, _canonical_hash, _read_json, _relative, _require
from .joint_interventions_cli import (
    ARMS,
    CONTRAST_FIELDS,
    OUTCOME_FIELDS,
    SUBJECT_FIELDS,
    JointPlan,
    JointSubject,
    _load_plan,
    _read_csv,
    _starting_states,
    _unit_id,
    _unit_plan_raw,
    _unit_record,
)
from .paired_interventions_cli import COMPARISONS, VARIANTS
from .parameters import mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .states import STATE_NAMES


AGREEMENT_FIELDS = (
    'batch', 'unit_id', 'kind', 'cell_id', 'split', 'arm', 'comparison',
    'passes', 'reason', 'trajectory_maximum', 'trajectory_worst_state',
    'outcomes_maximum', 'outcomes_not_applicable_count',
)
INITIAL_FIELDS = (
    *SUBJECT_FIELDS, *STATE_NAMES, 'p14_batch_directory',
    'p14_batch_summary_sha256', 'p14_decision_sha256',
    'p14_main_run_sha256', 'p14_main_trajectory_sha256',
)
DESCRIPTIVE_FIELDS = (
    'arm', 'checkpoint', 'metric', 'n_defined', 'n_undefined',
    'minimum_percent_change', 'median_percent_change', 'maximum_percent_change',
)


def _agreement_row(batch: int, report: dict, arm: str,
                   label: str, check: dict) -> dict:
    unit = mapping(report['unit'], 'unit')
    trajectory = check.get('trajectory', {})
    outcomes = check.get('outcomes', {})
    return {
        'batch': batch, 'unit_id': unit['unit_id'], 'kind': unit['kind'],
        'cell_id': unit['cell_id'], 'split': unit['split'], 'arm': arm,
        'comparison': label, 'passes': check.get('passes'),
        'reason': check.get('reason'),
        'trajectory_maximum': trajectory.get('maximum'),
        'trajectory_worst_state': trajectory.get('worst_state'),
        'outcomes_maximum': outcomes.get('maximum'),
        'outcomes_not_applicable_count': outcomes.get('not_applicable_count'),
    }


def _batch_record(
    plan: JointPlan, batch: int, directory: Path, batch_one: dict | None,
) -> tuple[dict, dict, list[dict], list[dict], list[dict], dict[str, dict], str]:
    _require(directory.parent == plan.output_root / f'batch_{batch:02d}',
             f'JT1 batch {batch}: directory is outside its assigned root.')
    summary_raw = (directory / 'summary.json').read_bytes()
    summary = _read_json(directory / 'summary.json')
    inputs = _read_json(directory / 'inputs.json')
    subjects = plan.select(batch)
    units = (*plan.final_plan.gate, *subjects) if batch == 1 else subjects
    expected_integrations = len(units) * len(ARMS) * len(VARIANTS)
    expected_checks = len(units) * len(ARMS) * len(COMPARISONS)
    expected_outcomes = len(subjects) * len(ARMS) * 32
    expected_contrasts = len(subjects) * (len(ARMS) - 1) * 32
    _require(summary.get('profile') == inputs.get('profile') == plan.profile
             and summary.get('status') == 'confirmed'
             and summary.get('numerically_confirmed') is True
             and summary.get('issues') == []
             and summary.get('batch') == inputs.get('batch') == batch
             and summary.get('batch_count') == inputs.get('batch_count') == plan.batch_count
             and summary.get('batch_size') == plan.batch_size
             and summary.get('subjects_planned') == plan.batch_size
             and summary.get('subjects_attempted') == plan.batch_size
             and summary.get('subjects_numerically_confirmed') == plan.batch_size
             and summary.get('subjects_rejected') == 0
             and summary.get('arms_per_subject') == len(ARMS)
             and summary.get('variants_per_arm') == len(VARIANTS)
             and summary.get('integrations_planned') == expected_integrations
             and summary.get('integrations_attempted') == expected_integrations
             and summary.get('comparison_checks_planned') == expected_checks
             and summary.get('comparison_checks_passing') == expected_checks
             and summary.get('outcome_rows') == expected_outcomes
             and summary.get('contrast_rows') == expected_contrasts,
             f'JT1 batch {batch}: not fully confirmed under A2FULL.')
    if batch == 1:
        _require(summary.get('joint_gate_local_points') == 9
                 and summary.get('joint_gate_points_confirmed') == 9
                 and summary.get('joint_gate_confirmed') is True
                 and summary.get('joint_gate_source') is None
                 and inputs.get('joint_gate_source') is None,
                 'JT1 batch 1 does not confirm all nine gate points.')
    else:
        _require(batch_one is not None
                 and summary.get('joint_gate_local_points') == 0
                 and summary.get('joint_gate_points_confirmed') == 0
                 and summary.get('joint_gate_confirmed') is True
                 and summary.get('joint_gate_source') == inputs.get('joint_gate_source')
                 == {'directory': batch_one['directory'],
                     'summary_sha256': batch_one['summary_sha256']},
                 f'JT1 batch {batch}: gate lineage differs from batch 1.')
    for key, expected in (
        ('config_sha256', plan.config_hash),
        ('interventions_sha256', plan.catalog_hash),
        ('p14_summary_sha256', plan.p14_summary_hash),
        ('p12_summary_sha256', plan.p12_summary_hash),
        ('allocation_manifest_sha256', plan.allocation_hash),
        ('subject_manifest_sha256', plan.subject_hash),
    ):
        _require(summary.get(key) == inputs.get(key) == expected,
                 f'JT1 batch {batch}: {key} mismatch.')
    unit_raw = _unit_plan_raw(units)
    _require(summary.get('unit_plan_sha256') == inputs.get('unit_plan_sha256')
             == sha256(unit_raw)
             and (directory / 'unit_plan.csv').read_bytes() == unit_raw
             and inputs.get('unit_ids') == [_unit_id(unit) for unit in units]
             and inputs.get('subject_ids') == [row.candidate.candidate_id for row in subjects]
             and inputs.get('joint_gate_ids') == (
                 [row.candidate_id for row in plan.final_plan.gate] if batch == 1 else [])
             and sha256((directory / 'interventions_joint_final.yaml').read_bytes())
                 == plan.config_hash
             and sha256((directory / 'interventions.yaml').read_bytes()) == plan.catalog_hash
             and sha256((directory / 'final.yaml').read_bytes()) == plan.final_plan.config_hash
             and sha256((directory / 'p14_summary.json').read_bytes()) == plan.p14_summary_hash
             and sha256((directory / 'p12_summary.json').read_bytes()) == plan.p12_summary_hash
             and (directory / 'subject_allocation.csv').read_bytes() == plan.allocation_raw
             and (directory / 'subjects.csv').read_bytes() == plan.subject_raw,
             f'JT1 batch {batch}: archived plans or manifests changed.')
    source_hashes = inputs.get('source_sha256')
    _require(isinstance(source_hashes, dict)
             and source_hashes == summary.get('source_sha256'),
             f'JT1 batch {batch}: source hashes are inconsistent.')
    for name, digest in source_hashes.items():
        _require(sha256((directory / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'JT1 batch {batch}: source snapshot changed: {name}.')
    expected_settings = {
        name: _canonical_hash(asdict(value))
        for name, value in plan.final_plan.settings.items()
    }
    _require(inputs.get('settings_sha256') == summary.get('settings_sha256')
             == expected_settings,
             f'JT1 batch {batch}: numerical settings changed.')

    starting = _starting_states(plan, units)
    _require(inputs.get('starting_states') == starting,
             f'JT1 batch {batch}: P14 starting-state lineage changed.')
    agreements: list[dict] = []
    tabular_outcomes: list[dict] = []
    tabular_contrasts: list[dict] = []
    for unit in units:
        unit_id = _unit_id(unit)
        report = _read_json(directory / 'cases' / unit_id / 'decision.json')
        candidate = unit.candidate if isinstance(unit, JointSubject) else unit
        case = _candidate_case(plan.final_plan, candidate)
        _require(report.get('unit') == _unit_record(unit)
                 and report.get('parameters') == asdict(case.parameters)
                 and report.get('subject') == asdict(case.subject)
                 and report.get('external_input') == asdict(case.external_input)
                 and report.get('starting_state') == starting[unit_id]
                 and report.get('numerically_confirmed') is True
                 and report.get('accepted') is True
                 and report.get('issues') == []
                 and report.get('integrations_attempted') == len(ARMS) * len(VARIANTS),
                 f'JT1 batch {batch}/{unit_id}: decision is not confirmed.')
        comparisons = mapping(report.get('comparisons'), f'{unit_id} comparisons')
        _require(set(comparisons) == set(ARMS),
                 f'JT1 batch {batch}/{unit_id}: arm coverage changed.')
        initial = [starting[unit_id]['state'][name] for name in STATE_NAMES]
        for arm_name in ARMS:
            checks = mapping(comparisons[arm_name], f'{unit_id}/{arm_name} comparisons')
            _require(set(checks) == {label for label, _, _ in COMPARISONS}
                     and all(check.get('passes') is True for check in checks.values()),
                     f'JT1 batch {batch}/{unit_id}/{arm_name}: checks are incomplete.')
            agreements.extend(_agreement_row(batch, report, arm_name, label, checks[label])
                              for label, _, _ in COMPARISONS)
            for variant in VARIANTS:
                run_dir = directory / 'cases' / unit_id / arm_name / variant
                run = _read_json(run_dir / 'run.json')
                metadata = mapping(run.get('metadata'), f'{unit_id}/{arm_name}/{variant} metadata')
                _require(run.get('status') == report['arms'][arm_name][variant]['status']
                         == 'completed'
                         and run.get('settings_sha256') == expected_settings[variant]
                         and run.get('trajectory_file') == 'trajectory.npz'
                         and (run_dir / 'trajectory.npz').is_file()
                         and (run_dir / 'outcomes.json').is_file()
                         and metadata.get('parameters') == asdict(case.parameters)
                         and metadata.get('subject') == asdict(case.subject)
                         and metadata.get('input') ==
                             {'kind': 'ConstantInput', **asdict(case.external_input)}
                         and metadata.get('intervention') == asdict(plan.arms[arm_name])
                         and metadata.get('initial_state') == initial,
                         f'JT1 batch {batch}/{unit_id}/{arm_name}/{variant}: archive incomplete.')
        case_outcomes = _read_csv(directory / 'cases' / unit_id / 'outcomes.csv',
                                  OUTCOME_FIELDS)
        case_contrasts = _read_csv(
            directory / 'cases' / unit_id / 'paired_contrasts.csv', CONTRAST_FIELDS)
        _require(len(case_outcomes) == len(ARMS) * 32
                 and len(case_contrasts) == (len(ARMS) - 1) * 32,
                 f'JT1 batch {batch}/{unit_id}: per-unit tabular coverage changed.')
        if isinstance(unit, JointSubject):
            tabular_outcomes.extend(case_outcomes)
            tabular_contrasts.extend(case_contrasts)
    _require(len(agreements) == expected_checks,
             f'JT1 batch {batch}: agreement coverage changed.')

    outcomes = _read_csv(directory / 'outcomes.csv', OUTCOME_FIELDS)
    contrasts = _read_csv(directory / 'paired_contrasts.csv', CONTRAST_FIELDS)
    subject_ids = {row.candidate.candidate_id for row in subjects}
    _require(len(outcomes) == expected_outcomes
             and len(contrasts) == expected_contrasts
             and outcomes == tabular_outcomes
             and contrasts == tabular_contrasts
             and {row['unit_id'] for row in outcomes} == subject_ids
             and {row['unit_id'] for row in contrasts} == subject_ids
             and all(int(row['batch']) == batch for row in outcomes + contrasts)
             and len({(row['unit_id'], row['arm'], row['checkpoint'], row['metric'])
                      for row in outcomes}) == expected_outcomes
             and len({(row['unit_id'], row['arm'], row['checkpoint'], row['metric'])
                      for row in contrasts}) == expected_contrasts,
             f'JT1 batch {batch}: aggregate tabular coverage changed.')
    return summary, inputs, outcomes, contrasts, agreements, starting, sha256(summary_raw)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _initial_rows(plan: JointPlan, starting_by_id: dict[str, dict]) -> list[dict]:
    rows = []
    for subject in plan.subjects:
        subject_id = subject.candidate.candidate_id
        start = starting_by_id[subject_id]
        rows.append({
            **subject.record(),
            **start['state'],
            'p14_batch_directory': start['p14_batch_directory'],
            'p14_batch_summary_sha256': start['p14_batch_summary_sha256'],
            'p14_decision_sha256': start['p14_decision_sha256'],
            'p14_main_run_sha256': start['p14_main_run_sha256'],
            'p14_main_trajectory_sha256': start['p14_main_trajectory_sha256'],
        })
    return rows


def _descriptive_rows(contrasts: list[dict]) -> list[dict]:
    output = []
    for arm in ARMS[1:]:
        for checkpoint in (200., 500., 1000., 2000.):
            for metric in ('AUC_A42', 'A42_final', 'Ratio_42_40', 'AUC_O',
                           'P_final', 'AUC_I', 'C99_max', 'AUC_C99'):
                subset = [row for row in contrasts
                          if row['arm'] == arm
                          and float(row['checkpoint']) == checkpoint
                          and row['metric'] == metric]
                _require(len(subset) == 1024,
                         f'{arm}/{metric}@{checkpoint}: expected 1024 paired rows.')
                values = [float(row['percent_change_from_control']) for row in subset
                          if row['percent_change_from_control'] != '']
                _require(all(isfinite(value) for value in values),
                         f'{arm}/{metric}@{checkpoint}: nonfinite contrast.')
                output.append({
                    'arm': arm, 'checkpoint': checkpoint, 'metric': metric,
                    'n_defined': len(values), 'n_undefined': 1024 - len(values),
                    'minimum_percent_change': min(values) if values else None,
                    'median_percent_change': median(values) if values else None,
                    'maximum_percent_change': max(values) if values else None,
                })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Collect eight confirmed P15 JT1-A2FULL batches without simulation.')
    parser.add_argument('--config', type=Path,
                        default=Path('config/interventions_joint_final.yaml'))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--from-root', action='store_true')
    selection.add_argument('--batch-dirs', type=Path, nargs='+')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate archives without RHS, integration or writes.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = _load_plan(args.config)
        _require(0 < args.progress_interval < float('inf'),
                 '--progress-interval must be positive and finite.')
        if args.from_root:
            matches = [list((plan.output_root / f'batch_{number:02d}').glob('*/summary.json'))
                       for number in range(1, plan.batch_count + 1)]
            _require(all(len(items) == 1 for items in matches),
                     'Exactly one completed JT1 directory per batch is required; use --batch-dirs.')
            directories = [items[0].parent.resolve() for items in matches]
        else:
            directories = [path.resolve() for path in args.batch_dirs]
        _require(len(directories) == plan.batch_count
                 and len(set(directories)) == plan.batch_count,
                 'Supply eight distinct JT1 batch directories in batch order.')
    except (OSError, UnicodeError, KeyError, TypeError, ValueError,
            yaml.YAMLError) as error:
        parser.error(str(error))

    progress = None if args.no_progress else ProgressBar(
        plan.batch_count, label='P15 JT1 cierre', unit='lotes',
        interval=args.progress_interval)
    if progress is not None:
        progress.start()
    outcomes: list[dict] = []
    contrasts: list[dict] = []
    agreements: list[dict] = []
    starts_by_id: dict[str, dict] = {}
    batches: dict[str, dict] = {}
    source_hashes = None
    settings_hashes = None
    batch_one = None
    try:
        for number, directory in enumerate(directories, start=1):
            if progress is not None:
                progress.stage(f'lote {number}: verificando')
            summary, inputs, batch_outcomes, batch_contrasts, batch_agreements, starts, digest = (
                _batch_record(plan, number, directory, batch_one))
            if source_hashes is None:
                source_hashes = inputs['source_sha256']
                settings_hashes = inputs['settings_sha256']
            _require(inputs['source_sha256'] == source_hashes
                     and inputs['settings_sha256'] == settings_hashes,
                     f'JT1 batch {number}: source or settings differ across batches.')
            record = {'directory': _relative(directory), 'summary_sha256': digest}
            batches[str(number)] = record
            if number == 1:
                batch_one = record
            outcomes.extend(batch_outcomes)
            contrasts.extend(batch_contrasts)
            agreements.extend(batch_agreements)
            starts_by_id.update({key: value for key, value in starts.items()
                                 if key.startswith('J1_') and not key.startswith('J1_GATE_')})
            if progress is not None:
                progress.advance(f'lote {number}/{plan.batch_count}: confirmado')
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    finally:
        if progress is not None:
            progress.close()

    _require(len(outcomes) == 262144
             and len(contrasts) == 229376
             and len(agreements) == 16528
             and len(starts_by_id) == 1024
             and all(row['passes'] is True for row in agreements),
             'JT1 collection lost outcome, contrast, state or agreement coverage.')
    initial_rows = _initial_rows(plan, starts_by_id)
    descriptive = _descriptive_rows(contrasts)
    if args.check_only:
        print('P15 JT1 collection preflight OK: 8/8 lotes, 1024/1024 sujetos, '
              '24792/24792 integraciones, 16528/16528 acuerdos.')
        print('262144 outcomes y 229376 contrastes tratados verificados.')
        print('Sin RHS, integración ni archivos nuevos.')
        return 0

    output = plan.output_root / 'collection' / datetime.now(timezone.utc).strftime(
        '%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'interventions_joint_final.yaml').write_bytes(plan.raw)
    (output / 'interventions.yaml').write_bytes(plan.catalog_raw)
    (output / 'final.yaml').write_bytes(plan.final_config_raw)
    (output / 'p14_summary.json').write_bytes(plan.p14_summary_raw)
    (output / 'p12_summary.json').write_bytes(plan.p12_summary_raw)
    (output / 'subject_allocation.csv').write_bytes(plan.allocation_raw)
    (output / 'subjects.csv').write_bytes(plan.subject_raw)
    _write_csv(output / 'initial_states.csv', INITIAL_FIELDS, initial_rows)
    _write_csv(output / 'outcomes.csv', OUTCOME_FIELDS, outcomes)
    _write_csv(output / 'paired_contrasts.csv', CONTRAST_FIELDS, contrasts)
    _write_csv(output / 'numerical_agreement.csv', AGREEMENT_FIELDS, agreements)
    _write_csv(output / 'descriptive_summary.csv', DESCRIPTIVE_FIELDS, descriptive)
    summary = {
        'profile': plan.profile, 'status': 'confirmed',
        'numerically_confirmed': True,
        'technical_acceptance_only': True,
        'biological_status': 'unclassified',
        'population_inference': False,
        'responder_classification': 'not_assessed',
        'subjects_selected': 1024, 'cells': 64,
        'subjects_per_cell': 16, 'batches_confirmed': 8,
        'batch_size': 128, 'arms_per_subject': len(ARMS),
        'variants_per_arm': len(VARIANTS),
        'joint_gate_points_confirmed': 9,
        'integrations_archived': 24792,
        'agreement_rows': len(agreements),
        'agreement_rows_passing': len(agreements),
        'outcome_rows': len(outcomes), 'contrast_rows': len(contrasts),
        'initial_state_rows': len(initial_rows),
        'descriptive_rows': len(descriptive),
        'config_sha256': plan.config_hash,
        'interventions_sha256': plan.catalog_hash,
        'p14_summary_sha256': plan.p14_summary_hash,
        'p14_candidates_sha256': plan.p14_candidates_hash,
        'p14_outcomes_sha256': plan.p14_outcomes_hash,
        'p12_summary_sha256': plan.p12_summary_hash,
        'p12_contrasts_sha256': plan.p12_contrasts_hash,
        'p12_descriptive_sha256': plan.p12_descriptive_hash,
        'allocation_manifest_sha256': plan.allocation_hash,
        'subject_manifest_sha256': plan.subject_hash,
        'derived_seeds': plan.seeds,
        'source_sha256': source_hashes, 'settings_sha256': settings_hashes,
        'batches': batches,
        'subjects_file': 'subjects.csv',
        'initial_states_file': 'initial_states.csv',
        'outcomes_file': 'outcomes.csv',
        'contrasts_file': 'paired_contrasts.csv',
        'agreement_file': 'numerical_agreement.csv',
        'descriptive_file': 'descriptive_summary.csv',
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    print('confirmed: 8/8 lotes, 1024/1024 sujetos, '
          '24792/24792 integraciones, 16528/16528 acuerdos')
    print(f'Artifacts: {output}')
    return 0
