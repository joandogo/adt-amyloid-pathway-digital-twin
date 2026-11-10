"""Collect ten confirmed P14 J1+A2 batches without integrating."""

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
from .final_cohort_cli import (
    ARCHIVED_METRICS,
    COMPARISONS,
    VARIANTS,
    FinalCohortPlan,
    _canonical_hash,
    _candidate_csv,
    _candidate_case,
    _load_plan,
    _read_json,
    _require,
    _relative,
)
from .parameters import mapping
from .progress import ProgressBar
from .reference_artifacts import sha256


OUTCOME_FIELDS = (
    'candidate_id', 'index', 'accepted', 'k_BACE_multiplier', 'C_i', 'I_ext',
    'checkpoint', 'metric', 'value', 'reason', 'sample_time',
)
AGREEMENT_FIELDS = (
    'batch', 'candidate_id', 'kind', 'comparison', 'passes', 'reason',
    'trajectory_maximum', 'trajectory_worst_state',
    'outcomes_maximum', 'outcomes_not_applicable_count',
)


def _read_outcomes(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == OUTCOME_FIELDS,
                 f'{path}: unexpected outcome columns.')
        return list(reader)


def _agreement_row(batch: int, report: dict, label: str, check: dict) -> dict:
    trajectory = check.get('trajectory', {})
    outcomes = check.get('outcomes', {})
    return {
        'batch': batch,
        'candidate_id': report['candidate_id'],
        'kind': report['kind'],
        'comparison': label,
        'passes': check.get('passes'),
        'reason': check.get('reason'),
        'trajectory_maximum': trajectory.get('maximum'),
        'trajectory_worst_state': trajectory.get('worst_state'),
        'outcomes_maximum': outcomes.get('maximum'),
        'outcomes_not_applicable_count': outcomes.get('not_applicable_count'),
    }


def _batch_record(plan: FinalCohortPlan, batch: int, directory: Path,
                  batch_one: dict | None) -> tuple[dict, dict, list[dict], list[dict], str]:
    _require(directory.parent == plan.output_root / f'batch_{batch:02d}',
             f'Batch {batch}: directory is outside its assigned root.')
    summary_raw = (directory / 'summary.json').read_bytes()
    summary = _read_json(directory / 'summary.json')
    inputs = _read_json(directory / 'inputs.json')
    candidates = plan.select(batch)
    units = (*plan.gate, *candidates) if batch == 1 else candidates
    expected_integrations = len(units) * len(VARIANTS)
    expected_checks = len(units) * len(COMPARISONS)
    _require(summary.get('profile') == inputs.get('profile') == plan.profile
             and summary.get('status') == 'confirmed'
             and summary.get('numerically_confirmed') is True
             and summary.get('issues') == []
             and summary.get('batch') == inputs.get('batch') == batch
             and summary.get('batch_count') == inputs.get('batch_count') == plan.batch_count
             and summary.get('batch_size') == plan.batch_size
             and summary.get('seed') == inputs.get('seed') == plan.seed
             and summary.get('candidates_planned') == plan.batch_size
             and summary.get('candidates_attempted') == plan.batch_size
             and summary.get('candidates_numerically_confirmed') == plan.batch_size
             and summary.get('candidates_rejected') == 0
             and summary.get('integrations_planned') == expected_integrations
             and summary.get('integrations_attempted') == expected_integrations
             and summary.get('variants_per_candidate') == len(VARIANTS)
             and summary.get('comparison_checks_planned') == expected_checks
             and summary.get('comparison_checks_passing') == expected_checks
             and summary.get('primary_variant') == inputs.get('primary_variant') == 'main'
             and summary.get('trajectory_archive') == inputs.get('trajectory_archive')
                 == 'npz_compressed_all_variants',
             f'Batch {batch}: not fully confirmed under J1+A2.')
    if batch == 1:
        _require(summary.get('joint_gate_local_points') == 9
                 and summary.get('joint_gate_points_confirmed') == 9
                 and summary.get('joint_gate_confirmed') is True
                 and summary.get('joint_gate_source') is None
                 and inputs.get('joint_gate_source') is None,
                 'Batch 1 does not confirm all nine joint-gate points.')
    else:
        _require(batch_one is not None
                 and summary.get('joint_gate_local_points') == 0
                 and summary.get('joint_gate_points_confirmed') == 0
                 and summary.get('joint_gate_confirmed') is True
                 and summary.get('joint_gate_source') == inputs.get('joint_gate_source')
                 == {'directory': batch_one['directory'],
                     'summary_sha256': batch_one['summary_sha256']},
                 f'Batch {batch}: joint-gate lineage differs from selected batch 1.')
    for key, expected in (
        ('config_sha256', plan.config_hash),
        ('pilot_summary_sha256', plan.pilot_summary_hash),
        ('pilot_inputs_sha256', plan.pilot_inputs_hash),
        ('sobol_c_summary_sha256', plan.sobol_c_summary_hash),
    ):
        _require(summary.get(key) == inputs.get(key) == expected,
                 f'Batch {batch}: {key} mismatch.')
    expected_plan = _candidate_csv(units)
    _require(summary.get('unit_plan_sha256') == inputs.get('unit_plan_sha256')
             == sha256(expected_plan)
             and (directory / 'unit_plan.csv').read_bytes() == expected_plan
             and inputs.get('unit_ids') == [row.candidate_id for row in units]
             and inputs.get('candidate_ids') == [row.candidate_id for row in candidates]
             and inputs.get('joint_gate_ids') == (
                 [row.candidate_id for row in plan.gate] if batch == 1 else [])
             and sha256((directory / 'final.yaml').read_bytes()) == plan.config_hash
             and sha256((directory / 'pilot_summary.json').read_bytes()) == plan.pilot_summary_hash
             and sha256((directory / 'sobol_c_summary.json').read_bytes())
                 == plan.sobol_c_summary_hash,
             f'Batch {batch}: archived plan, configuration or lineage changed.')
    source_hashes = inputs.get('source_sha256')
    _require(isinstance(source_hashes, dict)
             and source_hashes == summary.get('source_sha256'),
             f'Batch {batch}: source hashes are missing or inconsistent.')
    for name, digest in source_hashes.items():
        _require(sha256((directory / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'Batch {batch}: source snapshot changed: {name}.')
    settings_hashes = inputs.get('settings_sha256')
    expected_settings = {
        name: _canonical_hash(asdict(value))
        for name, value in plan.settings.items()
    }
    _require(isinstance(settings_hashes, dict) and settings_hashes == expected_settings,
             f'Batch {batch}: numerical settings differ from pinned C1 controls.')

    agreements: list[dict] = []
    for candidate in units:
        case_dir = directory / 'cases' / candidate.candidate_id
        report = _read_json(case_dir / 'decision.json')
        case = _candidate_case(plan, candidate)
        _require(all(report.get(key) == value for key, value in candidate.record().items())
                 and report.get('parameters') == asdict(case.parameters)
                 and report.get('subject') == asdict(case.subject)
                 and report.get('external_input') == asdict(case.external_input)
                 and report.get('numerically_confirmed') is True
                 and report.get('accepted') is True
                 and report.get('issues') == []
                 and report.get('integrations_attempted') == len(VARIANTS),
                 f'Batch {batch}/{candidate.candidate_id}: decision is not confirmed.')
        checks = report.get('comparisons')
        _require(isinstance(checks, dict)
                 and set(checks) == {label for label, _, _ in COMPARISONS}
                 and all(check.get('passes') is True for check in checks.values()),
                 f'Batch {batch}/{candidate.candidate_id}: numerical checks are incomplete.')
        agreements.extend(_agreement_row(batch, report, label, checks[label])
                          for label, _, _ in COMPARISONS)
        for variant in VARIANTS:
            run_dir = case_dir / variant
            run = _read_json(run_dir / 'run.json')
            metadata = mapping(run.get('metadata'), 'archived run metadata')
            _require(run.get('status') == report['runs'][variant]['status'] == 'completed'
                     and run.get('settings_sha256') == settings_hashes[variant]
                     and run.get('trajectory_file') == 'trajectory.npz'
                     and (run_dir / 'trajectory.npz').is_file()
                     and (run_dir / 'outcomes.json').is_file()
                     and metadata.get('parameters') == asdict(case.parameters)
                     and metadata.get('subject') == asdict(case.subject)
                     and metadata.get('input') ==
                         {'kind': 'ConstantInput', **asdict(case.external_input)}
                     and metadata.get('intervention') ==
                         {'kind': 'control', 'strength': None}
                     and metadata.get('initial_state') == list(map(float, plan.initial_state)),
                     f'Batch {batch}/{candidate.candidate_id}/{variant}: archive is incomplete.')
    _require(len(agreements) == expected_checks,
             f'Batch {batch}: numerical-agreement coverage differs.')

    outcomes = _read_outcomes(directory / 'outcomes.csv')
    expected_rows = plan.batch_size * len(plan.checkpoints) * len(ARCHIVED_METRICS)
    expected_keys = {(candidate.candidate_id, checkpoint, metric)
                     for candidate in candidates for checkpoint in plan.checkpoints
                     for metric in ARCHIVED_METRICS}
    actual_keys = {(row['candidate_id'], float(row['checkpoint']), row['metric'])
                   for row in outcomes}
    expected_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    _require(len(outcomes) == expected_rows and actual_keys == expected_keys
             and len(actual_keys) == len(outcomes)
             and all(row['accepted'] == 'True'
                     and int(row['index']) == expected_by_id[row['candidate_id']].index
                     and float(row['k_BACE_multiplier']) ==
                         expected_by_id[row['candidate_id']].k_BACE_multiplier
                     and float(row['C_i']) == expected_by_id[row['candidate_id']].C_i
                     and float(row['I_ext']) == expected_by_id[row['candidate_id']].I_ext
                     for row in outcomes),
             f'Batch {batch}: main-outcome coverage differs.')
    return summary, inputs, outcomes, agreements, sha256(summary_raw)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Collect ten confirmed P14 J1+A2 batches without simulation.')
    parser.add_argument('--config', type=Path, default=Path('config/cohorts/final.yaml'))
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
                     'Exactly one completed P14 directory per batch is required; use --batch-dirs.')
            directories = [items[0].parent.resolve() for items in matches]
        else:
            directories = [path.resolve() for path in args.batch_dirs]
        _require(len(directories) == plan.batch_count
                 and len(set(directories)) == plan.batch_count,
                 'Supply ten distinct P14 batch directories in batch order.')
    except (OSError, UnicodeError, KeyError, TypeError, ValueError,
            yaml.YAMLError) as error:
        parser.error(str(error))

    progress = None if args.no_progress else ProgressBar(
        plan.batch_count, label='P14 cierre', unit='lotes',
        interval=args.progress_interval)
    if progress is not None:
        progress.start()
    outcomes: list[dict] = []
    agreements: list[dict] = []
    batches: dict[str, dict] = {}
    source_hashes = None
    settings_hashes = None
    batch_one = None
    try:
        for number, directory in enumerate(directories, start=1):
            if progress is not None:
                progress.stage(f'lote {number}: verificando')
            summary, inputs, batch_outcomes, batch_agreements, digest = _batch_record(
                plan, number, directory, batch_one)
            if source_hashes is None:
                source_hashes = inputs['source_sha256']
                settings_hashes = inputs['settings_sha256']
            _require(inputs['source_sha256'] == source_hashes
                     and inputs['settings_sha256'] == settings_hashes,
                     f'Batch {number}: source or numerical settings differ across batches.')
            record = {'directory': _relative(directory), 'summary_sha256': digest}
            batches[str(number)] = record
            if number == 1:
                batch_one = record
            outcomes.extend({'batch': number, **row} for row in batch_outcomes)
            agreements.extend(batch_agreements)
            if progress is not None:
                progress.advance(f'lote {number}/{plan.batch_count}: confirmado')
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    finally:
        if progress is not None:
            progress.close()

    expected_outcomes = plan.target * len(plan.checkpoints) * len(ARCHIVED_METRICS)
    expected_agreements = (plan.target + len(plan.gate)) * len(COMPARISONS)
    _require(len(outcomes) == expected_outcomes
             and len(agreements) == expected_agreements
             and all(row['passes'] is True for row in agreements),
             'P14 collection lost outcome or numerical-agreement coverage.')
    descriptive = []
    for checkpoint in plan.checkpoints:
        for metric in ARCHIVED_METRICS:
            subset = [row for row in outcomes
                      if float(row['checkpoint']) == checkpoint and row['metric'] == metric]
            _require(len(subset) == plan.target,
                     f'{metric}@{checkpoint}: expected all 10000 design points.')
            values = [float(row['value']) for row in subset if row['value'] != '']
            _require(all(isfinite(value) for value in values),
                     f'{metric}@{checkpoint}: nonfinite archived value.')
            _require(metric == 'Ratio_42_40' or len(values) == plan.target,
                     f'{metric}@{checkpoint}: confirmed cohort has undefined required values.')
            descriptive.append({
                'checkpoint': checkpoint, 'metric': metric,
                'n_defined': len(values), 'n_undefined': plan.target - len(values),
                'minimum': min(values) if values else None,
                'median': median(values) if values else None,
                'maximum': max(values) if values else None,
            })
    if args.check_only:
        print('P14 collection preflight OK: 10/10 lotes, 10000/10000 candidatos, '
              '30027/30027 integraciones archivadas, 20018/20018 acuerdos.')
        print('Sin RHS, integración ni archivos nuevos.')
        return 0

    output = plan.output_root / 'collection' / datetime.now(timezone.utc).strftime(
        '%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'final.yaml').write_bytes(plan.raw)
    (output / 'pilot_summary.json').write_bytes(plan.pilot_summary_raw)
    (output / 'sobol_c_summary.json').write_bytes(plan.sobol_c_summary_raw)
    (output / 'candidates.csv').write_bytes(_candidate_csv(plan.candidates))
    _write_csv(output / 'outcomes.csv', ('batch', *OUTCOME_FIELDS), outcomes)
    _write_csv(output / 'numerical_agreement.csv', AGREEMENT_FIELDS, agreements)
    _write_csv(output / 'descriptive_summary.csv',
               ('checkpoint', 'metric', 'n_defined', 'n_undefined',
                'minimum', 'median', 'maximum'), descriptive)
    summary = {
        'profile': plan.profile, 'status': 'confirmed',
        'numerically_confirmed': True,
        'technical_acceptance_only': True,
        'biological_status': 'unclassified',
        'population_inference': False,
        'synthetic_independent_uniform_design': True,
        'seed': plan.seed, 'batches_confirmed': plan.batch_count,
        'batch_size': plan.batch_size,
        'candidates_planned': plan.target,
        'candidates_accepted': plan.target,
        'candidates_rejected': 0,
        'joint_gate_points_confirmed': len(plan.gate),
        'integrations_archived': 30027,
        'variants_per_candidate': len(VARIANTS),
        'primary_variant': 'main',
        'trajectory_archive': 'npz_compressed_all_variants',
        'agreement_rows': len(agreements),
        'agreement_rows_passing': len(agreements),
        'outcome_rows': len(outcomes),
        'descriptive_rows': len(descriptive),
        'config_sha256': plan.config_hash,
        'pilot_summary_sha256': plan.pilot_summary_hash,
        'pilot_inputs_sha256': plan.pilot_inputs_hash,
        'sobol_c_summary_sha256': plan.sobol_c_summary_hash,
        'source_sha256': source_hashes,
        'settings_sha256': settings_hashes,
        'batches': batches,
        'candidates_file': 'candidates.csv',
        'outcomes_file': 'outcomes.csv',
        'agreement_file': 'numerical_agreement.csv',
        'descriptive_file': 'descriptive_summary.csv',
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    print('confirmed: 10/10 lotes, 10000/10000 candidatos, '
          '30027/30027 integraciones archivadas, 20018/20018 acuerdos')
    print(f'Artifacts: {output}')
    return 0
