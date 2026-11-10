"""Run the selected P14 J1+A2 synthetic cohort in resumable batches."""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from .basal_cli import _json
from .inputs import ConstantInput
from .outcomes import AgreementSettings
from .parameters import (
    ModelParameters,
    SubjectParameters,
    finite_number,
    load_config,
    mapping,
)
from .progress import ProgressBar
from .reference_artifacts import sha256
from .simulate import SimulationSettings
from .states import STATE_NAMES
from .sweep_plan import SweepCase
from .sweeps import SweepRun, _compare as _compare_runs, _run


VARIANTS = ('main', 'radau_refined', 'bdf_refined')
COMPARISONS = (
    ('main_vs_radau_refined', 'main', 'radau_refined'),
    ('refined_methods', 'radau_refined', 'bdf_refined'),
)
DIMENSIONS = ('k_BACE_multiplier', 'C_i', 'I_ext')
ARCHIVED_METRICS = (
    'AUC_A42', 'A42_final', 'Ratio_42_40', 'AUC_O',
    'P_final', 'AUC_I', 'C99_max', 'AUC_C99',
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    return mapping(value, str(path))


def _read_json_value(path: Path) -> object:
    return json.loads(path.read_text(encoding='utf-8'))


def _pinned(path: Path, expected: str, label: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    _require(sha256(raw) == expected, f'{label} SHA-256 mismatch.')
    return raw, mapping(json.loads(raw), label)


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha256(raw)


@dataclass(frozen=True)
class CohortCandidate:
    candidate_id: str
    index: int
    kind: str
    normalized: tuple[float, float, float]
    k_BACE_multiplier: float
    C_i: float
    I_ext: float

    def record(self) -> dict[str, object]:
        return {
            'candidate_id': self.candidate_id,
            'index': self.index,
            'kind': self.kind,
            'normalized': dict(zip(DIMENSIONS, self.normalized)),
            'k_BACE_multiplier': self.k_BACE_multiplier,
            'C_i': self.C_i,
            'I_ext': self.I_ext,
        }


@dataclass(frozen=True)
class FinalCohortPlan:
    profile: str
    raw: bytes
    config_hash: str
    output_root: Path
    target: int
    batch_size: int
    batch_count: int
    seed: int
    ranges: dict[str, tuple[float, float]]
    candidates: tuple[CohortCandidate, ...]
    gate: tuple[CohortCandidate, ...]
    pilot_dir: Path
    pilot_summary_raw: bytes
    pilot_summary_hash: str
    pilot_inputs_hash: str
    sobol_c_dir: Path
    sobol_c_summary_raw: bytes
    sobol_c_summary_hash: str
    nominal: ModelParameters
    initial_state: np.ndarray
    settings: dict[str, SimulationSettings]
    agreement: AgreementSettings
    checkpoints: tuple[float, ...]

    def select(self, batch: int) -> tuple[CohortCandidate, ...]:
        if type(batch) is not int or not 1 <= batch <= self.batch_count:
            raise ValueError(f'Select a batch between 1 and {self.batch_count}.')
        start = (batch - 1) * self.batch_size
        return self.candidates[start:start + self.batch_size]


def _lhs_coordinates(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Generate a reproducible randomized Latin hypercube with PCG64."""
    _require(type(count) is int and count > 1 and type(dimensions) is int
             and dimensions > 0 and type(seed) is int and seed >= 0,
             'Latin-hypercube count, dimensions and seed are invalid.')
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty((count, dimensions), dtype=float)
    expected = np.arange(count)
    for dimension in range(dimensions):
        strata = rng.permutation(count)
        values[:, dimension] = (strata + rng.random(count)) / count
        actual = np.sort(np.floor(values[:, dimension] * count).astype(int))
        _require(np.array_equal(actual, expected),
                 'Latin-hypercube marginal lost or duplicated a stratum.')
    _require(np.all(np.isfinite(values)) and np.all((0 < values) & (values < 1)),
             'Latin-hypercube coordinates must be finite and strictly inside (0,1).')
    return values


def _scale(normalized: tuple[float, float, float],
           ranges: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
    return tuple(ranges[name][0] + value * (ranges[name][1] - ranges[name][0])
                 for name, value in zip(DIMENSIONS, normalized))


def _cohort_candidates(count: int, seed: int,
                       ranges: dict[str, tuple[float, float]]) -> tuple[CohortCandidate, ...]:
    coordinates = _lhs_coordinates(count, len(DIMENSIONS), seed)
    output = []
    for index, row in enumerate(coordinates, start=1):
        normalized = tuple(map(float, row))
        k_bace, c_i, i_ext = _scale(normalized, ranges)
        output.append(CohortCandidate(f'J1_{index:05d}', index, 'cohort', normalized,
                                      k_bace, c_i, i_ext))
    _require(len({row.candidate_id for row in output}) == count
             and len({(row.k_BACE_multiplier, row.C_i, row.I_ext)
                      for row in output}) == count,
             'J1 generated duplicate IDs or driver triples.')
    return tuple(output)


def _gate_candidates(ranges: dict[str, tuple[float, float]]) -> tuple[CohortCandidate, ...]:
    coordinates = [*itertools.product((0., 1.), repeat=3), (0.5, 0.5, 0.5)]
    output = []
    for index, normalized in enumerate(coordinates, start=1):
        k_bace, c_i, i_ext = _scale(normalized, ranges)
        output.append(CohortCandidate(f'J1_GATE_{index:03d}', index, 'joint_gate',
                                      normalized, k_bace, c_i, i_ext))
    _require(len(output) == 9 and len({(row.k_BACE_multiplier, row.C_i, row.I_ext)
                                      for row in output}) == 9,
             'Joint gate must contain eight distinct corners and one center.')
    return tuple(output)


def _candidate_csv(rows: tuple[CohortCandidate, ...]) -> bytes:
    stream = io.StringIO(newline='')
    fields = ('candidate_id', 'index', 'kind',
              'normalized_k_BACE_multiplier', 'normalized_C_i', 'normalized_I_ext',
              'k_BACE_multiplier', 'C_i', 'I_ext')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({
            'candidate_id': row.candidate_id, 'index': row.index, 'kind': row.kind,
            'normalized_k_BACE_multiplier': row.normalized[0],
            'normalized_C_i': row.normalized[1],
            'normalized_I_ext': row.normalized[2],
            'k_BACE_multiplier': row.k_BACE_multiplier,
            'C_i': row.C_i, 'I_ext': row.I_ext,
        })
    return stream.getvalue().encode('utf-8')


def _load_plan(path: Path) -> FinalCohortPlan:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _require(set(data) == {'profile', 'status', 'source', 'state_order', 'intervention',
                           'cohort', 'joint_gate', 'initialization', 'evaluation',
                           'numerics', 'acceptance', 'outcomes', 'output_root'},
             'Unexpected stage-14 configuration fields.')
    _require(data['profile'] == 'P14_J1_A2' and data['status'] == 'selected'
             and data['state_order'] == list(STATE_NAMES)
             and data['intervention'] == 'control',
             'Stage 14 requires selected P14_J1_A2 and the v0.3 control state order.')

    source = mapping(data['source'], 'source')
    _require(set(source) == {'pilot_dir', 'pilot_summary_sha256', 'pilot_inputs_sha256',
                             'sobol_c_collection', 'sobol_c_summary_sha256'},
             'Stage-14 source lineage is incomplete.')
    pilot_dir = Path(source['pilot_dir']).resolve()
    pilot_summary_raw, pilot_summary = _pinned(
        pilot_dir / 'summary.json', source['pilot_summary_sha256'], 'C1 pilot summary')
    _require(pilot_dir.is_relative_to(Path('results/cohorts/pilot').resolve())
             and pilot_summary.get('profile') == 'P11_C1'
             and pilot_summary.get('status') == 'confirmed'
             and pilot_summary.get('numerically_confirmed') is True
             and pilot_summary.get('candidates_accepted') == 120
             and pilot_summary.get('candidates_rejected') == 0
             and pilot_summary.get('issues') == []
             and pilot_summary.get('inputs_sha256') == source['pilot_inputs_sha256'],
             'Stage 14 requires the pinned fully confirmed C1 pilot.')
    pilot_inputs_raw, pilot_inputs = _pinned(
        pilot_dir / 'inputs.json', source['pilot_inputs_sha256'], 'C1 pilot inputs')
    pilot_config_raw = (pilot_dir / 'pilot.yaml').read_bytes()
    _require(sha256(pilot_config_raw) == pilot_summary.get('config_sha256')
             == pilot_inputs.get('config_sha256')
             and pilot_inputs.get('state_order') == list(STATE_NAMES),
             'Pinned C1 configuration, inputs or state order changed.')
    pilot_config = mapping(yaml.safe_load(pilot_config_raw.decode('utf-8')), 'C1 config')

    sobol_c_dir = Path(source['sobol_c_collection']).resolve()
    sobol_c_summary_raw, sobol_c = _pinned(
        sobol_c_dir / 'summary.json', source['sobol_c_summary_sha256'], 'Sobol C summary')
    _require(sobol_c_dir.is_relative_to(
                Path('results/sensitivity/sobol_c/collection').resolve())
             and sobol_c.get('profile') == 'P13_Sobol_C'
             and sobol_c.get('status') == 'confirmed'
             and sobol_c.get('numerically_confirmed') is True
             and sobol_c.get('batches_confirmed') == 4
             and sobol_c.get('agreement_rows') == 3072
             and sobol_c.get('agreement_rows_passing') == 3072
             and sobol_c.get('integrations_combined_archived') == 4608
             and sobol_c.get('sobol_a_indices_defined') == 32
             and sobol_c.get('sobol_a_indices_undefined') == 4,
             'Stage 14 requires the pinned confirmed Sobol A+C collection.')

    cohort = mapping(data['cohort'], 'cohort')
    _require(set(cohort) == {'design', 'generator', 'dimensions', 'target_candidates',
                             'target_accepted', 'driver_ranges', 'independence', 'seed',
                             'seed_role', 'batch_size', 'batch_count',
                             'replacement_policy', 'reference_policy'},
             'Stage-14 cohort specification is incomplete.')
    _require(cohort['design'] == 'randomized_latin_hypercube'
             and cohort['generator'] == 'numpy_pcg64_axis_permutations_with_uniform_jitter'
             and cohort['dimensions'] == list(DIMENSIONS)
             and cohort['target_candidates'] == cohort['target_accepted'] == 10000
             and cohort['independence'] == 'synthetic_design_assumption'
             and cohort['seed'] == 1103
             and cohort['seed_role'] == 'technical_reproducibility_only'
             and cohort['batch_size'] == 1000 and cohort['batch_count'] == 10
             and cohort['replacement_policy'] == 'no_replacement'
             and cohort['reference_policy'] == 'reuse_confirmed_c1_baseline_not_counted',
             'Stage-14 cohort differs from selected J1.')
    ranges_raw = mapping(cohort['driver_ranges'], 'driver_ranges')
    _require(set(ranges_raw) == set(DIMENSIONS), 'J1 requires exactly three driver ranges.')
    ranges = {}
    for name in DIMENSIONS:
        entry = mapping(ranges_raw[name], name)
        _require(set(entry) == {'lower', 'upper'}, f'{name} requires lower and upper.')
        lower = finite_number(entry['lower'], f'{name}.lower')
        upper = finite_number(entry['upper'], f'{name}.upper')
        _require(lower < upper, f'{name} requires an increasing range.')
        ranges[name] = (lower, upper)
    expected_ranges = {'k_BACE_multiplier': (1.25, 2.),
                       'C_i': (0.5, 0.75), 'I_ext': (0.25, 1.)}
    _require(ranges == expected_ranges
             and mapping(mapping(pilot_config['cohort'], 'C1 cohort')['driver_ranges'],
                         'C1 driver ranges') == ranges_raw,
             'J1 must preserve the selected C1 design bounds exactly.')

    gate = mapping(data['joint_gate'], 'joint_gate')
    _require(gate == {'enabled': True, 'design': 'full_factorial_corners_plus_center',
                      'points': 9, 'assigned_batch': 1,
                      'required_before_candidates': True},
             'J1 requires the selected nine-point joint numerical gate.')
    _require(mapping(data['initialization'], 'initialization') ==
             {'protocol': 'common_basal_step'},
             'Stage 14 requires common_basal_step initialization.')
    evaluation = mapping(data['evaluation'], 'evaluation')
    _require(evaluation == {'horizon': 2000.,
                            'checkpoints': [200., 500., 1000., 2000.]},
             'Stage-14 evaluation must preserve C1 horizon and checkpoints.')
    numerics = mapping(data['numerics'], 'numerics')
    _require(numerics == {'variants': list(VARIANTS), 'source': 'pinned_pilot_c1',
                          'trajectory_archive': 'npz_compressed_all_variants'},
             'Stage 14 requires A2 and compressed trajectories for all variants.')
    acceptance = mapping(data['acceptance'], 'acceptance')
    _require(acceptance == {
        'mode': 'technical_only', 'require_full_trajectory': True,
        'require_numerical_agreement_per_candidate': True,
        'allowed_undefined_outcome': 'A40_zero_at_checkpoint_for_Ratio_42_40_only',
        'failure_policy': 'no_silent_replacement', 'biological_status': 'unclassified',
    }, 'Stage-14 acceptance differs from selected A2 technical acceptance.')
    outcomes = mapping(data['outcomes'], 'outcomes')
    _require(outcomes == {'primary_variant': 'main',
                          'archived_metrics': list(ARCHIVED_METRICS),
                          'threshold_time_enabled': False,
                          'population_inference': False},
             'Stage-14 outcomes differ from the selected control-only cohort.')

    cases = mapping(pilot_inputs.get('cases'), 'C1 cases')
    baseline = mapping(cases.get('baseline'), 'C1 baseline case')
    nominal = ModelParameters(**mapping(baseline.get('parameters'), 'C1 parameters'))
    _require(nominal.n_O == 2. and nominal.n_P == 1.5,
             'Stage 14 keeps nominal P1 n_O/n_P; C did not authorize subject-level orders.')
    initial = mapping(pilot_inputs.get('initial_state'), 'C1 initial state')
    _require(set(initial) == set(STATE_NAMES), 'C1 initial state is incomplete.')
    initial_state = np.asarray([finite_number(initial[name], name) for name in STATE_NAMES],
                               dtype=float)
    settings_raw = mapping(pilot_inputs.get('settings'), 'C1 settings')
    _require(set(settings_raw) == set(VARIANTS), 'C1 numerical variants changed.')
    settings = {name: SimulationSettings(**mapping(settings_raw[name], name))
                for name in VARIANTS}
    _require(settings['main'].method == 'Radau'
             and settings['radau_refined'].method == 'Radau'
             and settings['bdf_refined'].method == 'BDF'
             and all(item.t_span == (0., 2000.) for item in settings.values())
             and all(value in settings['main'].t_eval for value in evaluation['checkpoints']),
             'Pinned C1 numerical settings do not cover selected A2.')
    agreement = AgreementSettings(**mapping(pilot_inputs.get('agreement'), 'C1 agreement'))

    output_root = Path(data['output_root']).resolve()
    results_root = Path('results').resolve()
    _require(data['output_root'] == 'results/cohorts/final'
             and output_root != results_root and output_root.is_relative_to(results_root),
             'Stage-14 output must remain under results/cohorts/final.')
    candidates = _cohort_candidates(cohort['target_candidates'], cohort['seed'], ranges)
    gate_candidates = _gate_candidates(ranges)
    _require(len(candidates) == cohort['batch_size'] * cohort['batch_count'],
             'J1 candidate count does not match its ten batches.')
    return FinalCohortPlan(
        data['profile'], raw, sha256(raw), output_root, cohort['target_candidates'],
        cohort['batch_size'], cohort['batch_count'], cohort['seed'], ranges,
        candidates, gate_candidates, pilot_dir, pilot_summary_raw,
        source['pilot_summary_sha256'], source['pilot_inputs_sha256'], sobol_c_dir,
        sobol_c_summary_raw, source['sobol_c_summary_sha256'], nominal, initial_state,
        settings, agreement, tuple(evaluation['checkpoints']),
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _gate_source(plan: FinalCohortPlan, batch: int,
                 explicit: Path | None) -> dict[str, str] | None:
    if batch == 1:
        _require(explicit is None, '--gate-dir is only valid for batches 2–10.')
        return None
    if explicit is None:
        matches = list((plan.output_root / 'batch_01').glob('*/summary.json'))
        _require(len(matches) == 1,
                 'Exactly one completed batch-1 directory is required; use --gate-dir.')
        directory = matches[0].parent.resolve()
    else:
        directory = explicit.resolve()
    _require(directory.parent == plan.output_root / 'batch_01',
             '--gate-dir must be a completed batch-1 directory.')
    raw = (directory / 'summary.json').read_bytes()
    summary = _read_json(directory / 'summary.json')
    _require(summary.get('profile') == plan.profile
             and summary.get('status') == 'confirmed'
             and summary.get('batch') == 1
             and summary.get('config_sha256') == plan.config_hash
             and summary.get('joint_gate_confirmed') is True
             and summary.get('joint_gate_points_confirmed') == 9
             and summary.get('candidates_numerically_confirmed') == plan.batch_size,
             'Selected batch 1 does not confirm the joint gate and its 1000 candidates.')
    inputs = _read_json(directory / 'inputs.json')
    current_hashes = {name: sha256(content) for name, content in _source_files().items()}
    current_settings = {name: _canonical_hash(asdict(value))
                        for name, value in plan.settings.items()}
    _require(summary.get('source_sha256') == inputs.get('source_sha256') == current_hashes
             and inputs.get('settings_sha256') == current_settings,
             'Current source or numerical settings differ from confirmed batch 1.')
    for name, digest in current_hashes.items():
        _require(sha256((directory / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'Batch-1 source snapshot changed: {name}.')
    return {'directory': _relative(directory), 'summary_sha256': sha256(raw)}


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


def _save_progress(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _save_run(output: Path, candidate_id: str, variant: str,
              run: SweepRun, settings_hash: str) -> None:
    directory = output / 'cases' / candidate_id / variant
    directory.mkdir(parents=True)
    record = {'status': run.status, 'message': run.message,
              'variant': variant, 'settings_sha256': settings_hash}
    if run.result is not None:
        result = run.result
        if result.t.ndim == 1 and result.y.shape == (len(STATE_NAMES), len(result.t)):
            trajectory = directory / 'trajectory.npz'
            with trajectory.open('xb') as stream:
                np.savez_compressed(stream, tau=result.t, states=result.y,
                                    state_order=np.asarray(STATE_NAMES))
            record.update(trajectory_file='trajectory.npz', trajectory_format='numpy_npz_compressed',
                          trajectory_samples=int(len(result.t)))
        else:
            record['trajectory_note'] = 'Returned arrays have an invalid shape.'
        if run.status == 'completed':
            metadata = dict(result.metadata)
            metadata.pop('settings', None)
            record.update(nfev=int(result.nfev), njev=int(result.njev), nlu=int(result.nlu),
                          minimum_state=float(np.min(result.y)),
                          final_time=float(result.t[-1]),
                          final_state=dict(zip(STATE_NAMES, map(float, result.y[:, -1]))),
                          metadata=metadata)
    else:
        record['trajectory_note'] = 'Integrator returned no usable trajectory.'
    _json(directory / 'run.json', record)
    _json(directory / 'outcomes.json', run.outcomes)


def _candidate_case(plan: FinalCohortPlan, candidate: CohortCandidate) -> SweepCase:
    return SweepCase(
        candidate.candidate_id,
        replace(plan.nominal, k_BACE=plan.nominal.k_BACE * candidate.k_BACE_multiplier),
        SubjectParameters(candidate.C_i),
        ConstantInput(candidate.I_ext),
    )


def _candidate_report(plan: FinalCohortPlan, candidate: CohortCandidate,
                      case: SweepCase, runs: dict[str, SweepRun]) -> dict:
    comparisons = {label: _compare_runs(runs[left], runs[right], plan)
                   for label, left, right in COMPARISONS}
    undefined = [
        {'checkpoint': row['checkpoint'], 'metric': row['metric'], 'reason': row['reason']}
        for row in runs['main'].outcomes if row['value'] is None
        and not (row['metric'] == 'Ratio_42_40'
                 and row['reason'] == 'A40_zero_at_checkpoint')
    ]
    issues = [f'{name}: {run.message}' for name, run in runs.items()
              if run.status != 'completed']
    issues.extend(f'{label}: numerical_agreement_failed'
                  for label, result in comparisons.items() if result['passes'] is not True)
    issues.extend(f"undefined {row['metric']} at {row['checkpoint']}" for row in undefined)
    return {
        **candidate.record(),
        'parameters': asdict(case.parameters),
        'subject': asdict(case.subject),
        'external_input': asdict(case.external_input),
        'runs': {name: {'status': run.status, 'message': run.message}
                 for name, run in runs.items()},
        'comparisons': comparisons,
        'undefined_outcomes': undefined,
        'integrations_attempted': len(VARIANTS),
        'numerically_confirmed': not issues,
        'accepted': not issues,
        'issues': issues,
    }


def _outcome_rows(candidate: CohortCandidate, accepted: bool,
                  outcomes: object) -> list[dict]:
    _require(isinstance(outcomes, list), f'{candidate.candidate_id}: outcomes must be a list.')
    rows = []
    for raw in outcomes:
        row = mapping(raw, f'{candidate.candidate_id} outcome')
        if row.get('metric') in ARCHIVED_METRICS:
            rows.append({
                'candidate_id': candidate.candidate_id,
                'index': candidate.index,
                'accepted': accepted,
                'k_BACE_multiplier': candidate.k_BACE_multiplier,
                'C_i': candidate.C_i,
                'I_ext': candidate.I_ext,
                'checkpoint': row.get('checkpoint'),
                'metric': row.get('metric'),
                'value': row.get('value'),
                'reason': row.get('reason'),
                'sample_time': row.get('sample_time'),
            })
    _require(len(rows) == len(plan_checkpoints := (200., 500., 1000., 2000.))
             * len(ARCHIVED_METRICS)
             and {row['checkpoint'] for row in rows} == set(plan_checkpoints),
             f'{candidate.candidate_id}: main outcomes lack selected metric coverage.')
    return rows


def _write_outcomes(path: Path, rows: list[dict]) -> None:
    fields = ('candidate_id', 'index', 'accepted', 'k_BACE_multiplier', 'C_i', 'I_ext',
              'checkpoint', 'metric', 'value', 'reason', 'sample_time')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _resume_completed(plan: FinalCohortPlan, batch: int, output: Path,
                      candidates: tuple[CohortCandidate, ...], gate_source: dict | None,
                      current_hashes: dict[str, str]) -> tuple[dict[str, dict], list[dict]]:
    expected_parent = plan.output_root / f'batch_{batch:02d}'
    _require(output.parent == expected_parent and output.is_dir(),
             '--resume-dir must be an existing run of the selected batch.')
    _require(not (output / 'summary.json').exists() and not (output / 'outcomes.csv').exists(),
             'This batch already has final artifacts and cannot be resumed.')
    archived = _read_json(output / 'inputs.json')
    units = (*plan.gate, *candidates) if batch == 1 else candidates
    plan_raw = _candidate_csv(units)
    _require(sha256((output / 'final.yaml').read_bytes()) == plan.config_hash
             and sha256((output / 'pilot_summary.json').read_bytes()) == plan.pilot_summary_hash
             and sha256((output / 'sobol_c_summary.json').read_bytes()) == plan.sobol_c_summary_hash
             and sha256((output / 'unit_plan.csv').read_bytes()) == sha256(plan_raw),
             'Archived configuration, lineage or unit plan changed.')
    _require(archived.get('profile') == plan.profile
             and archived.get('batch') == batch
             and archived.get('batch_count') == plan.batch_count
             and archived.get('config_sha256') == plan.config_hash
             and archived.get('pilot_summary_sha256') == plan.pilot_summary_hash
             and archived.get('pilot_inputs_sha256') == plan.pilot_inputs_hash
             and archived.get('sobol_c_summary_sha256') == plan.sobol_c_summary_hash
             and archived.get('unit_plan_sha256') == sha256(plan_raw)
             and archived.get('unit_ids') == [row.candidate_id for row in units]
             and archived.get('joint_gate_source') == gate_source,
             'Resume lineage, batch assignment or gate source differs.')
    old_hashes = mapping(archived.get('source_sha256'), 'archived source hashes')
    _require(old_hashes == current_hashes,
             'Source changed after this batch started; preserve it and start a new attempt.')
    for name, digest in old_hashes.items():
        _require(sha256((output / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'Archived implementation changed: {name}.')
    settings_hashes = mapping(archived.get('settings_sha256'), 'settings hashes')
    _require(
        settings_hashes
        == {
            name: _canonical_hash(asdict(value))
            for name, value in plan.settings.items()
        },
        'Archived numerical settings differ from pinned C1 controls.',
    )
    reports: dict[str, dict] = {}
    rows: list[dict] = []
    gap = False
    for candidate in units:
        case_dir = output / 'cases' / candidate.candidate_id
        decision = case_dir / 'decision.json'
        if not decision.exists():
            _require(not case_dir.exists(),
                     f'{candidate.candidate_id}: partial files exist; preserve this attempt and start another.')
            gap = True
            continue
        _require(not gap, 'Completed units must form a contiguous gate/candidate prefix.')
        report = _read_json(decision)
        case = _candidate_case(plan, candidate)
        _require(all(report.get(key) == value for key, value in candidate.record().items())
                 and report.get('parameters') == asdict(case.parameters)
                 and report.get('subject') == asdict(case.subject)
                 and report.get('external_input') == asdict(case.external_input)
                 and report.get('integrations_attempted') == len(VARIANTS)
                 and isinstance(report.get('accepted'), bool)
                 and isinstance(report.get('issues'), list),
                 f'{candidate.candidate_id}: archived decision differs from J1+A2.')
        for variant in VARIANTS:
            run_dir = case_dir / variant
            run = _read_json(run_dir / 'run.json')
            _require(run.get('status') == report['runs'][variant]['status']
                     and run.get('settings_sha256') == settings_hashes[variant]
                     and (run_dir / 'outcomes.json').is_file(),
                     f'{candidate.candidate_id}/{variant}: archived run differs from decision.')
            if run['status'] == 'completed':
                metadata = mapping(run.get('metadata'), 'archived run metadata')
                _require(run.get('trajectory_file') == 'trajectory.npz'
                         and (run_dir / 'trajectory.npz').is_file()
                         and metadata.get('parameters') == asdict(case.parameters)
                         and metadata.get('subject') == asdict(case.subject)
                         and metadata.get('input') ==
                             {'kind': 'ConstantInput', **asdict(case.external_input)}
                         and metadata.get('intervention') ==
                             {'kind': 'control', 'strength': None}
                         and metadata.get('initial_state') == list(map(float, plan.initial_state)),
                         f'{candidate.candidate_id}/{variant}: completed archive lineage differs.')
        reports[candidate.candidate_id] = report
        if candidate.kind == 'cohort' and report['runs']['main']['status'] == 'completed':
            rows.extend(_outcome_rows(candidate, report['accepted'],
                        _read_json_value(case_dir / 'main' / 'outcomes.json')))
    return reports, rows


def _execute(plan: FinalCohortPlan, batch: int, candidates: tuple[CohortCandidate, ...],
             gate_source: dict | None, interval: float, show_progress: bool,
             resume_dir: Path | None) -> int:
    sources = _source_files()
    source_hashes = {name: sha256(raw) for name, raw in sources.items()}
    settings_hashes = {name: _canonical_hash(asdict(value))
                       for name, value in plan.settings.items()}
    units = (*plan.gate, *candidates) if batch == 1 else candidates
    unit_plan_raw = _candidate_csv(units)
    if resume_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = plan.output_root / f'batch_{batch:02d}' / stamp
        output.mkdir(parents=True, exist_ok=False)
        (output / 'final.yaml').write_bytes(plan.raw)
        (output / 'pilot_summary.json').write_bytes(plan.pilot_summary_raw)
        (output / 'sobol_c_summary.json').write_bytes(plan.sobol_c_summary_raw)
        (output / 'unit_plan.csv').write_bytes(unit_plan_raw)
        for name, raw in sources.items():
            copy = output / 'implementation' / 'adt' / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(raw)
        reports: dict[str, dict] = {}
        rows: list[dict] = []
        _json(output / 'inputs.json', {
            'profile': plan.profile, 'model_version': '0.3',
            'batch': batch, 'batch_count': plan.batch_count,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'config_sha256': plan.config_hash,
            'pilot_dir': _relative(plan.pilot_dir),
            'pilot_summary_sha256': plan.pilot_summary_hash,
            'pilot_inputs_sha256': plan.pilot_inputs_hash,
            'sobol_c_collection': _relative(plan.sobol_c_dir),
            'sobol_c_summary_sha256': plan.sobol_c_summary_hash,
            'source_sha256': source_hashes,
            'settings_sha256': settings_hashes,
            'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'PyYAML')},
            'state_order': STATE_NAMES,
            'initial_state': dict(zip(STATE_NAMES, map(float, plan.initial_state))),
            'seed': plan.seed, 'generator': 'numpy_pcg64_axis_permutations_with_uniform_jitter',
            'unit_plan_sha256': sha256(unit_plan_raw),
            'unit_ids': [row.candidate_id for row in units],
            'candidate_ids': [row.candidate_id for row in candidates],
            'joint_gate_ids': [row.candidate_id for row in plan.gate] if batch == 1 else [],
            'joint_gate_source': gate_source,
            'checkpoints': plan.checkpoints,
            'variants': VARIANTS,
            'trajectory_archive': 'npz_compressed_all_variants',
            'primary_variant': 'main',
        })
    else:
        output = resume_dir.resolve()
        reports, rows = _resume_completed(plan, batch, output, candidates,
                                          gate_source, source_hashes)

    total = len(units) * len(VARIANTS)
    progress = None if not show_progress else ProgressBar(
        total, label=f'P14 lote {batch}/{plan.batch_count}',
        unit='integraciones', interval=interval)
    if progress is not None:
        progress.done = len(reports) * len(VARIANTS)
        progress.start()

    def run_one(candidate: CohortCandidate) -> None:
        if candidate.candidate_id in reports:
            return
        case = _candidate_case(plan, candidate)
        runs = {}
        for variant in VARIANTS:
            phase = f'{candidate.candidate_id}/{variant}'
            if progress is not None:
                progress.stage(f'{phase}: integrando')
            run = _run(case, plan.initial_state, plan.settings[variant], plan.checkpoints)
            runs[variant] = run
            _save_run(output, candidate.candidate_id, variant, run, settings_hashes[variant])
            if progress is not None:
                progress.advance(f'{phase}: {run.status}; archivo guardado')
        report = _candidate_report(plan, candidate, case, runs)
        reports[candidate.candidate_id] = report
        _json(output / 'cases' / candidate.candidate_id / 'decision.json', report)
        if candidate.kind == 'cohort' and runs['main'].status == 'completed':
            rows.extend(_outcome_rows(candidate, report['accepted'], runs['main'].outcomes))
        candidate_reports = [reports[row.candidate_id] for row in candidates
                             if row.candidate_id in reports]
        gate_reports = [reports[row.candidate_id] for row in plan.gate
                        if row.candidate_id in reports]
        _save_progress(output / 'progress.json', {
            'last_unit': candidate.candidate_id,
            'joint_gate_attempted': len(gate_reports),
            'joint_gate_confirmed': sum(row['accepted'] for row in gate_reports),
            'candidates_attempted': len(candidate_reports),
            'candidates_accepted': sum(row['accepted'] for row in candidate_reports),
            'candidates_rejected': sum(not row['accepted'] for row in candidate_reports),
            'integrations_attempted': len(reports) * len(VARIANTS),
        })

    try:
        if batch == 1:
            for candidate in plan.gate:
                run_one(candidate)
            gate_ok = all(reports[row.candidate_id]['accepted'] for row in plan.gate)
            if gate_ok:
                for candidate in candidates:
                    run_one(candidate)
        else:
            gate_ok = True
            for candidate in candidates:
                run_one(candidate)
    finally:
        if progress is not None:
            progress.close()

    _write_outcomes(output / 'outcomes.csv', rows)
    candidate_reports = [reports[row.candidate_id] for row in candidates
                         if row.candidate_id in reports]
    accepted = sum(row['accepted'] for row in candidate_reports)
    rejected = sum(not row['accepted'] for row in candidate_reports)
    complete = len(candidate_reports) == len(candidates)
    confirmed = gate_ok and complete and accepted == len(candidates)
    status = 'confirmed' if confirmed else ('gate_not_confirmed' if not gate_ok
                                             else 'not_confirmed')
    issues = [f'{candidate_id}: {issue}' for candidate_id, report in reports.items()
              for issue in report['issues']]
    summary = {
        'profile': plan.profile, 'status': status,
        'numerically_confirmed': confirmed,
        'technical_acceptance_only': True,
        'biological_status': 'unclassified',
        'population_inference': False,
        'batch': batch, 'batch_count': plan.batch_count,
        'seed': plan.seed, 'batch_size': plan.batch_size,
        'candidates_planned': len(candidates),
        'candidates_attempted': len(candidate_reports),
        'candidates_numerically_confirmed': accepted,
        'candidates_rejected': rejected,
        'joint_gate_required': True,
        'joint_gate_local_points': len(plan.gate) if batch == 1 else 0,
        'joint_gate_points_confirmed': (
            sum(reports[row.candidate_id]['accepted'] for row in plan.gate)
            if batch == 1 else 0),
        'joint_gate_confirmed': gate_ok,
        'joint_gate_source': gate_source,
        'integrations_planned': total,
        'integrations_attempted': len(reports) * len(VARIANTS),
        'variants_per_candidate': len(VARIANTS),
        'primary_variant': 'main',
        'trajectory_archive': 'npz_compressed_all_variants',
        'comparison_checks_planned': len(units) * len(COMPARISONS),
        'comparison_checks_passing': sum(
            check['passes'] is True for report in reports.values()
            for check in report['comparisons'].values()),
        'issues': issues,
        'config_sha256': plan.config_hash,
        'pilot_summary_sha256': plan.pilot_summary_hash,
        'pilot_inputs_sha256': plan.pilot_inputs_hash,
        'sobol_c_summary_sha256': plan.sobol_c_summary_hash,
        'source_sha256': source_hashes,
        'unit_plan_sha256': sha256(unit_plan_raw),
        'candidate_outcomes': 'outcomes.csv',
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    print(f'\n{status}: {accepted}/{len(candidates)} candidatos; '
          f'{summary["integrations_attempted"]}/{total} integraciones intentadas')
    print(f'Artifacts: {output}')
    return 0 if confirmed else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run one selected 1000-candidate P14 J1+A2 batch.')
    parser.add_argument('--config', type=Path, default=Path('config/cohorts/final.yaml'))
    parser.add_argument('--batch', type=int, required=True, help='Batch number, 1 through 10.')
    parser.add_argument('--gate-dir', type=Path,
                        help='Confirmed batch-1 directory when automatic selection is ambiguous.')
    parser.add_argument('--check-only', action='store_true',
                        help='Read-only preflight without RHS, integration or writes.')
    parser.add_argument('--resume-dir', type=Path,
                        help='Continue a contiguous prefix of complete units in this batch.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = _load_plan(args.config)
        candidates = plan.select(args.batch)
        _require(len(candidates) == plan.batch_size,
                 'Selected batch does not contain exactly 1000 candidates.')
        gate_source = _gate_source(plan, args.batch, args.gate_dir)
        _require(0 < args.progress_interval < float('inf'),
                 '--progress-interval must be positive and finite.')
        reused = 0
        if args.resume_dir is not None:
            hashes = {name: sha256(raw) for name, raw in _source_files().items()}
            reused = len(_resume_completed(plan, args.batch, args.resume_dir.resolve(),
                                           candidates, gate_source, hashes)[0])
    except (OSError, UnicodeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError, yaml.YAMLError) as error:
        parser.error(str(error))
    total_units = len(candidates) + (len(plan.gate) if args.batch == 1 else 0)
    if args.check_only:
        print(f'P14 preflight OK: lote {args.batch}/{plan.batch_count}, '
              f'{len(candidates)} candidatos, {total_units * len(VARIANTS)} integraciones; '
              f'{reused * len(VARIANTS)} reutilizables.')
        if args.batch == 1:
            print('La puerta conjunta de 8 esquinas + centro se ejecutará antes de los candidatos.')
        else:
            print(f'Puerta conjunta fijada por {gate_source["directory"]}.')
        print('Sin RHS, integración ni archivos nuevos.')
        return 0
    return _execute(plan, args.batch, candidates, gate_source,
                    args.progress_interval, not args.no_progress, args.resume_dir)
