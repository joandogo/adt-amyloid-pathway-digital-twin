"""Run the selected P15 JT1-A2FULL therapeutic trial in resumable batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from .basal_cli import _json
from .final_cohort_cli import (
    ARCHIVED_METRICS,
    CohortCandidate,
    FinalCohortPlan,
    _candidate_case,
    _candidate_csv,
    _canonical_hash,
    _load_plan as _load_final_plan,
    _read_json,
    _relative,
    _require,
)
from .model import state_vector
from .outcomes import contrast_outcomes
from .paired_interventions_cli import (
    COMPARISONS,
    VARIANTS,
    _arm_catalog,
    _compare,
    _run_arm,
)
from .parameters import finite_number, load_config, mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .states import STATE_NAMES
from .sweeps import SweepRun


ARMS = ('control', 'GSI_01', 'GSI_02', 'GSI_03', 'GSI_04',
        'GSM_low', 'GSM_medium', 'GSM_high')
DIMENSIONS = ('k_BACE_multiplier', 'C_i', 'I_ext')
SPLITS = ('train', 'validation', 'test')
SPLIT_COUNTS = {'train': 10, 'validation': 3, 'test': 3}
SELECTION_DOMAIN = 'P15_JT1_selection'
BATCH_DOMAIN = 'P15_JT1_batches'
SPLIT_DOMAIN = 'P15_E1_split'

OUTCOME_FIELDS = (
    'unit_id', 'kind', 'batch', 'cell_id', 'split', 'p14_index',
    'k_BACE_multiplier', 'C_i', 'I_ext', 'arm', 'intervention_kind',
    'strength', 'checkpoint', 'metric', 'value', 'reason', 'sample_time',
)
CONTRAST_FIELDS = (
    'unit_id', 'batch', 'cell_id', 'split', 'p14_index',
    'k_BACE_multiplier', 'C_i', 'I_ext', 'arm', 'intervention_kind',
    'strength', 'checkpoint', 'metric', 'is_primary', 'value',
    'control_value', 'difference_from_control',
    'percent_change_from_control', 'contrast_reason',
)


def _pinned_json(path: Path, expected: str, label: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    _require(sha256(raw) == expected, f'{label} SHA-256 mismatch.')
    return raw, mapping(json.loads(raw), label)


def _derived_seed(domain: str, lineage: str) -> int:
    """Derive a stable PCG64 integer from the first 128 SHA-256 bits."""
    digest = hashlib.sha256(f'{domain}|{lineage}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:16], byteorder='big', signed=False)


def _cell(candidate: CohortCandidate) -> tuple[int, int, int]:
    _require(all(0. <= value < 1. for value in candidate.normalized),
             f'{candidate.candidate_id}: normalized coordinates are outside [0,1).')
    bins = tuple(min(int(value * 4), 3) for value in candidate.normalized)
    _require(all(0 <= value < 4 for value in bins),
             f'{candidate.candidate_id}: normalized coordinates are outside [0,1].')
    return bins


def _cell_id(cell: tuple[int, int, int]) -> str:
    return 'C' + ''.join(str(value + 1) for value in cell)


@dataclass(frozen=True)
class SubjectAllocation:
    candidate: CohortCandidate
    cell: tuple[int, int, int]
    batch: int

    @property
    def cell_id(self) -> str:
        return _cell_id(self.cell)


@dataclass(frozen=True)
class JointSubject:
    candidate: CohortCandidate
    cell: tuple[int, int, int]
    batch: int
    split: str

    @property
    def cell_id(self) -> str:
        return _cell_id(self.cell)

    def record(self) -> dict[str, object]:
        return {
            'subject_id': self.candidate.candidate_id,
            'p14_index': self.candidate.index,
            'cell_id': self.cell_id,
            'bin_k_BACE_multiplier': self.cell[0] + 1,
            'bin_C_i': self.cell[1] + 1,
            'bin_I_ext': self.cell[2] + 1,
            'batch': self.batch,
            'split': self.split,
            'normalized_k_BACE_multiplier': self.candidate.normalized[0],
            'normalized_C_i': self.candidate.normalized[1],
            'normalized_I_ext': self.candidate.normalized[2],
            'k_BACE_multiplier': self.candidate.k_BACE_multiplier,
            'C_i': self.candidate.C_i,
            'I_ext': self.candidate.I_ext,
            'p14_source_batch': (self.candidate.index - 1) // 1000 + 1,
        }


ALLOCATION_FIELDS = (
    'subject_id', 'p14_index', 'cell_id', 'bin_k_BACE_multiplier',
    'bin_C_i', 'bin_I_ext', 'batch', 'normalized_k_BACE_multiplier',
    'normalized_C_i', 'normalized_I_ext', 'k_BACE_multiplier', 'C_i',
    'I_ext', 'p14_source_batch',
)
SUBJECT_FIELDS = (
    'subject_id', 'p14_index', 'cell_id', 'bin_k_BACE_multiplier',
    'bin_C_i', 'bin_I_ext', 'batch', 'split',
    'normalized_k_BACE_multiplier', 'normalized_C_i', 'normalized_I_ext',
    'k_BACE_multiplier', 'C_i', 'I_ext', 'p14_source_batch',
)


def _csv_bytes(fields: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _allocation_record(row: SubjectAllocation) -> dict[str, object]:
    candidate = row.candidate
    return {
        'subject_id': candidate.candidate_id,
        'p14_index': candidate.index,
        'cell_id': row.cell_id,
        'bin_k_BACE_multiplier': row.cell[0] + 1,
        'bin_C_i': row.cell[1] + 1,
        'bin_I_ext': row.cell[2] + 1,
        'batch': row.batch,
        'normalized_k_BACE_multiplier': candidate.normalized[0],
        'normalized_C_i': candidate.normalized[1],
        'normalized_I_ext': candidate.normalized[2],
        'k_BACE_multiplier': candidate.k_BACE_multiplier,
        'C_i': candidate.C_i,
        'I_ext': candidate.I_ext,
        'p14_source_batch': (candidate.index - 1) // 1000 + 1,
    }


def _subject_design(
    candidates: tuple[CohortCandidate, ...], p14_summary_hash: str,
) -> tuple[tuple[JointSubject, ...], bytes, bytes, dict[str, int]]:
    """Select, batch and split subjects without consulting any outcome."""
    grouped: dict[tuple[int, int, int], list[CohortCandidate]] = {}
    for candidate in sorted(candidates, key=lambda row: row.candidate_id):
        grouped.setdefault(_cell(candidate), []).append(candidate)
    expected_cells = {(a, b, c) for a in range(4) for b in range(4) for c in range(4)}
    _require(set(grouped) == expected_cells
             and all(len(rows) >= 16 for rows in grouped.values()),
             'P14 does not cover all 64 cells with at least 16 candidates.')

    selection_seed = _derived_seed(SELECTION_DOMAIN, p14_summary_hash)
    batch_seed = _derived_seed(BATCH_DOMAIN, p14_summary_hash)
    selection_rng = np.random.Generator(np.random.PCG64(selection_seed))
    batch_rng = np.random.Generator(np.random.PCG64(batch_seed))
    selected: dict[tuple[int, int, int], list[CohortCandidate]] = {}
    allocations: list[SubjectAllocation] = []
    for cell in sorted(expected_cells):
        eligible = grouped[cell]
        order = selection_rng.permutation(len(eligible))[:16]
        chosen = [eligible[int(index)] for index in order]
        selected[cell] = chosen
        batch_order = batch_rng.permutation(16)
        for batch in range(1, 9):
            for position in batch_order[(batch - 1) * 2:batch * 2]:
                allocations.append(SubjectAllocation(chosen[int(position)], cell, batch))
    allocations.sort(key=lambda row: (row.batch, row.cell, row.candidate.candidate_id))
    allocation_raw = _csv_bytes(
        ALLOCATION_FIELDS, [_allocation_record(row) for row in allocations])
    allocation_hash = sha256(allocation_raw)

    split_seed = _derived_seed(SPLIT_DOMAIN, allocation_hash)
    split_rng = np.random.Generator(np.random.PCG64(split_seed))
    split_by_id: dict[str, str] = {}
    labels = [name for name in SPLITS for _ in range(SPLIT_COUNTS[name])]
    for cell in sorted(expected_cells):
        cell_rows = sorted(selected[cell], key=lambda row: row.candidate_id)
        order = split_rng.permutation(16)
        for label, position in zip(labels, order):
            split_by_id[cell_rows[int(position)].candidate_id] = label
    subjects = tuple(JointSubject(row.candidate, row.cell, row.batch,
                                  split_by_id[row.candidate.candidate_id])
                     for row in allocations)
    subject_raw = _csv_bytes(SUBJECT_FIELDS, [row.record() for row in subjects])

    _require(len(subjects) == len({row.candidate.candidate_id for row in subjects}) == 1024,
             'JT1 selection must contain 1024 unique P14 subjects.')
    for cell in expected_cells:
        cell_subjects = [row for row in subjects if row.cell == cell]
        _require(len(cell_subjects) == 16
                 and {batch: sum(row.batch == batch for row in cell_subjects)
                      for batch in range(1, 9)} == {batch: 2 for batch in range(1, 9)}
                 and {split: sum(row.split == split for row in cell_subjects)
                      for split in SPLITS} == SPLIT_COUNTS,
                 f'{_cell_id(cell)}: allocation is not 16 subjects, 2/batch, 10/3/3.')
    return subjects, allocation_raw, subject_raw, {
        'selection': selection_seed, 'batch': batch_seed, 'split': split_seed,
    }


@dataclass(frozen=True)
class JointPlan:
    profile: str
    raw: bytes
    config_hash: str
    output_root: Path
    final_plan: FinalCohortPlan
    final_config_raw: bytes
    p14_dir: Path
    p14_summary_raw: bytes
    p14_summary_hash: str
    p14_candidates_hash: str
    p14_outcomes_hash: str
    p14_batches: dict[str, dict]
    p12_dir: Path
    p12_summary_raw: bytes
    p12_summary_hash: str
    p12_contrasts_hash: str
    p12_descriptive_hash: str
    catalog_raw: bytes
    catalog_hash: str
    arms: dict
    subjects: tuple[JointSubject, ...]
    allocation_raw: bytes
    subject_raw: bytes
    seeds: dict[str, int]
    checkpoints: tuple[float, ...]

    @property
    def allocation_hash(self) -> str:
        return sha256(self.allocation_raw)

    @property
    def subject_hash(self) -> str:
        return sha256(self.subject_raw)

    @property
    def batch_count(self) -> int:
        return 8

    @property
    def batch_size(self) -> int:
        return 128

    def select(self, batch: int) -> tuple[JointSubject, ...]:
        if type(batch) is not int or not 1 <= batch <= self.batch_count:
            raise ValueError('Select a JT1 batch between 1 and 8.')
        rows = tuple(row for row in self.subjects if row.batch == batch)
        _require(len(rows) == self.batch_size, f'JT1 batch {batch} must contain 128 subjects.')
        return rows


def _load_plan(path: Path) -> JointPlan:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _require(set(data) == {'profile', 'status', 'source', 'state_order', 'design',
                           'seeds', 'joint_gate', 'interventions', 'evaluation',
                           'outcomes', 'acceptance', 'archive', 'output_root'},
             'Unexpected stage-15 JT1 configuration fields.')
    _require(data['profile'] == 'P15_JT1_A2FULL' and data['status'] == 'selected'
             and data['state_order'] == list(STATE_NAMES),
             'JT1 requires selected P15_JT1_A2FULL and the v0.3 state order.')
    source = mapping(data['source'], 'source')
    _require(set(source) == {
        'final_config', 'final_config_sha256', 'p14_collection',
        'p14_summary_sha256', 'p14_candidates_sha256', 'p14_outcomes_sha256',
        'p12_collection', 'p12_summary_sha256', 'p12_contrasts_sha256',
        'p12_descriptive_sha256', 'interventions_file', 'interventions_sha256',
    }, 'JT1 source lineage is incomplete.')

    final_path = Path(source['final_config'])
    final_config_raw = final_path.read_bytes()
    _require(sha256(final_config_raw) == source['final_config_sha256'],
             'Pinned P14 configuration changed.')
    final_plan = _load_final_plan(final_path)
    _require(final_plan.config_hash == source['final_config_sha256'],
             'Regenerated P14 plan differs from its pinned configuration.')

    p14_dir = Path(source['p14_collection']).resolve()
    p14_summary_raw, p14 = _pinned_json(
        p14_dir / 'summary.json', source['p14_summary_sha256'], 'P14 collection')
    _require(p14_dir.is_relative_to(Path('results/cohorts/final/collection').resolve())
             and p14.get('profile') == 'P14_J1_A2'
             and p14.get('status') == 'confirmed'
             and p14.get('numerically_confirmed') is True
             and p14.get('candidates_accepted') == 10000
             and p14.get('candidates_rejected') == 0
             and p14.get('joint_gate_points_confirmed') == 9
             and p14.get('integrations_archived') == 30027
             and p14.get('agreement_rows') == 20018
             and p14.get('agreement_rows_passing') == 20018
             and p14.get('config_sha256') == final_plan.config_hash,
             'JT1 requires the pinned fully confirmed P14 collection.')
    candidate_raw = (p14_dir / 'candidates.csv').read_bytes()
    outcome_raw = (p14_dir / 'outcomes.csv').read_bytes()
    _require(sha256(candidate_raw) == source['p14_candidates_sha256']
             and candidate_raw == _candidate_csv(final_plan.candidates)
             and sha256(outcome_raw) == source['p14_outcomes_sha256'],
             'Pinned P14 candidate or outcome aggregate changed.')
    p14_batches = mapping(p14.get('batches'), 'P14 batches')
    _require(set(p14_batches) == {str(number) for number in range(1, 11)},
             'P14 collection must identify ten source batches.')
    for number in range(1, 11):
        record = mapping(p14_batches[str(number)], f'P14 batch {number}')
        directory = Path(record['directory']).resolve()
        summary_path = directory / 'summary.json'
        batch_raw = summary_path.read_bytes()
        batch_summary = mapping(json.loads(batch_raw), str(summary_path))
        _require(directory.parent == final_plan.output_root / f'batch_{number:02d}'
                 and sha256(batch_raw) == record.get('summary_sha256')
                 and batch_summary.get('status') == 'confirmed'
                 and batch_summary.get('batch') == number
                 and batch_summary.get('config_sha256') == final_plan.config_hash,
                 f'Pinned P14 batch {number} changed or is not confirmed.')

    p12_dir = Path(source['p12_collection']).resolve()
    p12_summary_raw, p12 = _pinned_json(
        p12_dir / 'summary.json', source['p12_summary_sha256'], 'P12 collection')
    _require(p12_dir.is_relative_to(Path('results/interventions/paired/collection').resolve())
             and p12.get('profile') == 'P12_C1_paired'
             and p12.get('status') == 'confirmed'
             and p12.get('numerically_confirmed') is True
             and p12.get('subjects_confirmed') == 120
             and p12.get('integrations_archived') == 2880
             and p12.get('contrast_rows') == 26880
             and p12.get('interventions_sha256') == source['interventions_sha256']
             and sha256((p12_dir / 'all_paired_contrasts.csv').read_bytes())
                 == source['p12_contrasts_sha256']
             and sha256((p12_dir / 'descriptive_summary.csv').read_bytes())
                 == source['p12_descriptive_sha256'],
             'JT1 requires the pinned confirmed P12 therapeutic protocol.')
    arms, catalog_raw = _arm_catalog(
        Path(source['interventions_file']), source['interventions_sha256'])
    _require(tuple(arms) == ARMS, 'JT1 intervention arm order changed.')

    design = mapping(data['design'], 'design')
    _require(design == {
        'source_candidates': 10000, 'selected_subjects': 1024,
        'stratification_dimensions': list(DIMENSIONS), 'bins_per_dimension': 4,
        'cells': 64, 'subjects_per_cell': 16,
        'selection': 'deterministic_without_replacement_within_cell',
        'batches': 8, 'batch_size': 128, 'subjects_per_cell_per_batch': 2,
        'numerical_policy': 'JT1-A2FULL', 'numerical_variants': list(VARIANTS),
        'initial_state': 'p14_main_final_state_at_2000_from_npz',
        'paired_control': 'own_subject_control',
        'e1_split_per_cell': SPLIT_COUNTS,
    }, 'JT1 design differs from the approved 1024-subject A2FULL protocol.')
    seeds = mapping(data['seeds'], 'seeds')
    _require(seeds == {
        'derivation': 'sha256_utf8_first_128_bits_big_endian_to_numpy_pcg64',
        'selection_domain': SELECTION_DOMAIN, 'batch_domain': BATCH_DOMAIN,
        'split_domain': SPLIT_DOMAIN, 'selection_lineage': 'p14_summary_sha256',
        'batch_lineage': 'p14_summary_sha256',
        'split_lineage': 'jt1_allocation_manifest_sha256',
    }, 'JT1 seed derivation changed.')
    _require(mapping(data['joint_gate'], 'joint_gate') == {
        'source': 'p14_joint_gate', 'points': 9, 'assigned_batch': 1,
        'required_before_subjects': True,
    } and mapping(data['interventions'], 'interventions') == {
        'arms': list(ARMS), 'combinations_enabled': False,
    }, 'JT1 gate or intervention catalog changed.')
    evaluation = mapping(data['evaluation'], 'evaluation')
    checkpoints = (200., 500., 1000., 2000.)
    _require(evaluation == {'horizon': 2000., 'checkpoints': list(checkpoints)},
             'JT1 horizon/checkpoints changed.')
    _require(mapping(data['outcomes'], 'outcomes') == {
        'primary_metric': 'AUC_A42', 'primary_checkpoint': 2000.,
        'contrast': 'percent_change_from_own_control',
        'archived_metrics': list(ARCHIVED_METRICS),
        'responder_classification': 'continuous_only',
        'biological_status': 'unclassified',
    }, 'JT1 outcome definitions changed.')
    _require(mapping(data['acceptance'], 'acceptance') == {
        'mode': 'technical_only', 'require_full_trajectory': True,
        'require_numerical_agreement_per_subject_arm': True,
        'allowed_undefined_outcome': 'A40_zero_at_checkpoint_for_Ratio_42_40_only',
        'failure_policy': 'no_silent_replacement', 'population_inference': False,
    } and mapping(data['archive'], 'archive') == {
        'trajectory': 'npz_compressed_all_variants', 'primary_variant': 'main',
    }, 'JT1 acceptance or archive policy changed.')
    _require(all(item.t_span == (0., 2000.) for item in final_plan.settings.values())
             and all(checkpoint in final_plan.settings['main'].t_eval
                     for checkpoint in checkpoints),
             'Pinned P14 numerical grids do not cover JT1.')
    subjects, allocation_raw, subject_raw, derived_seeds = _subject_design(
        final_plan.candidates, source['p14_summary_sha256'])
    output_root = Path(data['output_root']).resolve()
    results_root = Path('results').resolve()
    _require(data['output_root'] == 'results/interventions/joint_final'
             and output_root != results_root and output_root.is_relative_to(results_root),
             'JT1 output must remain under results/interventions/joint_final.')
    return JointPlan(
        data['profile'], raw, sha256(raw), output_root, final_plan,
        final_config_raw, p14_dir, p14_summary_raw, source['p14_summary_sha256'],
        source['p14_candidates_sha256'], source['p14_outcomes_sha256'], p14_batches,
        p12_dir, p12_summary_raw, source['p12_summary_sha256'],
        source['p12_contrasts_sha256'], source['p12_descriptive_sha256'],
        catalog_raw, source['interventions_sha256'], arms, subjects,
        allocation_raw, subject_raw, derived_seeds, checkpoints,
    )


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


def _source_directory(plan: JointPlan, candidate: CohortCandidate) -> tuple[int, Path, str]:
    number = 1 if candidate.kind == 'joint_gate' else (candidate.index - 1) // 1000 + 1
    record = mapping(plan.p14_batches[str(number)], f'P14 batch {number}')
    return number, Path(record['directory']).resolve(), str(record['summary_sha256'])


def _starting_state(plan: JointPlan, candidate: CohortCandidate) -> dict:
    """Load and audit one P14 main endpoint from its archived NPZ."""
    number, directory, batch_summary_hash = _source_directory(plan, candidate)
    case = _candidate_case(plan.final_plan, candidate)
    case_dir = directory / 'cases' / candidate.candidate_id
    decision_path = case_dir / 'decision.json'
    run_path = case_dir / 'main' / 'run.json'
    trajectory_path = case_dir / 'main' / 'trajectory.npz'
    decision_raw = decision_path.read_bytes()
    run_raw = run_path.read_bytes()
    decision = mapping(json.loads(decision_raw), str(decision_path))
    run = mapping(json.loads(run_raw), str(run_path))
    _require(all(decision.get(key) == value for key, value in candidate.record().items())
             and decision.get('accepted') is True
             and decision.get('numerically_confirmed') is True
             and decision.get('issues') == []
             and decision.get('parameters') == asdict(case.parameters)
             and decision.get('subject') == asdict(case.subject)
             and decision.get('external_input') == asdict(case.external_input),
             f'{candidate.candidate_id}: P14 decision differs from its confirmed case.')
    metadata = mapping(run.get('metadata'), f'{candidate.candidate_id} P14 metadata')
    final = mapping(run.get('final_state'), f'{candidate.candidate_id} P14 final state')
    _require(run.get('status') == 'completed' and run.get('final_time') == 2000.
             and run.get('trajectory_file') == 'trajectory.npz'
             and set(final) == set(STATE_NAMES)
             and metadata.get('parameters') == asdict(case.parameters)
             and metadata.get('subject') == asdict(case.subject)
             and metadata.get('input') == {'kind': 'ConstantInput', **asdict(case.external_input)}
             and metadata.get('intervention') == {'kind': 'control', 'strength': None},
             f'{candidate.candidate_id}: P14 main endpoint metadata changed.')
    final_vector = state_vector([
        finite_number(final[name], f'{candidate.candidate_id}.{name}')
        for name in STATE_NAMES
    ])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        _require(set(archive.files) == {'tau', 'states', 'state_order'},
                 f'{candidate.candidate_id}: P14 NPZ members changed.')
        tau = archive['tau']
        states = archive['states']
        order = tuple(str(value) for value in archive['state_order'])
        _require(tau.ndim == 1 and len(tau) > 1
                 and states.shape == (len(STATE_NAMES), len(tau))
                 and order == STATE_NAMES and float(tau[-1]) == 2000.
                 and np.array_equal(states[:, -1], final_vector),
                 f'{candidate.candidate_id}: P14 NPZ endpoint differs from run.json.')
    return {
        'state': dict(zip(STATE_NAMES, map(float, final_vector))),
        'p14_source_batch': number,
        'p14_batch_directory': _relative(directory),
        'p14_batch_summary_sha256': batch_summary_hash,
        'p14_decision_sha256': sha256(decision_raw),
        'p14_main_run_sha256': sha256(run_raw),
        'p14_main_trajectory_sha256': sha256(trajectory_path.read_bytes()),
    }


def _unit_candidate(unit: JointSubject | CohortCandidate) -> CohortCandidate:
    return unit.candidate if isinstance(unit, JointSubject) else unit


def _unit_id(unit: JointSubject | CohortCandidate) -> str:
    return _unit_candidate(unit).candidate_id


def _unit_kind(unit: JointSubject | CohortCandidate) -> str:
    return 'subject' if isinstance(unit, JointSubject) else 'joint_gate'


def _unit_record(unit: JointSubject | CohortCandidate) -> dict[str, object]:
    candidate = _unit_candidate(unit)
    return {
        'unit_id': candidate.candidate_id,
        'kind': _unit_kind(unit),
        'batch': unit.batch if isinstance(unit, JointSubject) else 1,
        'cell_id': unit.cell_id if isinstance(unit, JointSubject) else '',
        'split': unit.split if isinstance(unit, JointSubject) else '',
        'p14_index': candidate.index,
        'normalized_k_BACE_multiplier': candidate.normalized[0],
        'normalized_C_i': candidate.normalized[1],
        'normalized_I_ext': candidate.normalized[2],
        'k_BACE_multiplier': candidate.k_BACE_multiplier,
        'C_i': candidate.C_i,
        'I_ext': candidate.I_ext,
    }


UNIT_FIELDS = (
    'unit_id', 'kind', 'batch', 'cell_id', 'split', 'p14_index',
    'normalized_k_BACE_multiplier', 'normalized_C_i', 'normalized_I_ext',
    'k_BACE_multiplier', 'C_i', 'I_ext',
)


def _unit_plan_raw(units: tuple[JointSubject | CohortCandidate, ...]) -> bytes:
    return _csv_bytes(UNIT_FIELDS, [_unit_record(unit) for unit in units])


def _starting_states(
    plan: JointPlan, units: tuple[JointSubject | CohortCandidate, ...],
) -> dict[str, dict]:
    return {_unit_id(unit): _starting_state(plan, _unit_candidate(unit)) for unit in units}


def _gate_source(plan: JointPlan, batch: int, explicit: Path | None,
                 source_hashes: dict[str, str], settings_hashes: dict[str, str]) -> dict | None:
    if batch == 1:
        _require(explicit is None, '--gate-dir is only valid for batches 2–8.')
        return None
    if explicit is None:
        matches = list((plan.output_root / 'batch_01').glob('*/summary.json'))
        _require(len(matches) == 1,
                 'Exactly one completed JT1 batch-1 directory is required; use --gate-dir.')
        directory = matches[0].parent.resolve()
    else:
        directory = explicit.resolve()
    _require(directory.parent == plan.output_root / 'batch_01',
             '--gate-dir must identify a JT1 batch-1 directory.')
    raw = (directory / 'summary.json').read_bytes()
    summary = _read_json(directory / 'summary.json')
    inputs = _read_json(directory / 'inputs.json')
    _require(summary.get('profile') == plan.profile
             and summary.get('status') == 'confirmed'
             and summary.get('batch') == 1
             and summary.get('joint_gate_confirmed') is True
             and summary.get('joint_gate_points_confirmed') == 9
             and summary.get('subjects_numerically_confirmed') == plan.batch_size
             and summary.get('config_sha256') == plan.config_hash
             and inputs.get('source_sha256') == source_hashes
             and inputs.get('settings_sha256') == settings_hashes
             and inputs.get('subject_manifest_sha256') == plan.subject_hash,
             'Selected JT1 batch 1 does not confirm the gate and 128 subjects.')
    for name, digest in source_hashes.items():
        _require(sha256((directory / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'JT1 batch-1 source snapshot changed: {name}.')
    return {'directory': _relative(directory), 'summary_sha256': sha256(raw)}


def _save_run(output: Path, unit_id: str, arm: str, variant: str,
              run: SweepRun, settings_hash: str) -> None:
    directory = output / 'cases' / unit_id / arm / variant
    directory.mkdir(parents=True)
    record: dict[str, object] = {
        'status': run.status, 'message': run.message, 'arm': arm,
        'variant': variant, 'settings_sha256': settings_hash,
    }
    if run.result is not None:
        result = run.result
        if result.t.ndim == 1 and result.y.shape == (len(STATE_NAMES), len(result.t)):
            trajectory = directory / 'trajectory.npz'
            with trajectory.open('xb') as stream:
                np.savez_compressed(stream, tau=result.t, states=result.y,
                                    state_order=np.asarray(STATE_NAMES))
            record.update(trajectory_file='trajectory.npz',
                          trajectory_format='numpy_npz_compressed',
                          trajectory_samples=int(len(result.t)))
        else:
            record['trajectory_note'] = 'Returned arrays have an invalid shape.'
        if run.status == 'completed':
            metadata = dict(result.metadata)
            metadata.pop('settings', None)
            record.update(
                nfev=int(result.nfev), njev=int(result.njev), nlu=int(result.nlu),
                minimum_state=float(np.min(result.y)), final_time=float(result.t[-1]),
                final_state=dict(zip(STATE_NAMES, map(float, result.y[:, -1]))),
                metadata=metadata,
            )
    else:
        record['trajectory_note'] = 'Integrator returned no usable trajectory.'
    _json(directory / 'run.json', record)
    _json(directory / 'outcomes.json', run.outcomes)


def _outcome_rows(unit: JointSubject | CohortCandidate, arm_name: str,
                  intervention: object, outcomes: object) -> list[dict]:
    _require(isinstance(outcomes, list), f'{_unit_id(unit)}/{arm_name}: outcomes must be a list.')
    candidate = _unit_candidate(unit)
    base = _unit_record(unit)
    rows = []
    for raw in outcomes:
        row = mapping(raw, f'{candidate.candidate_id}/{arm_name} outcome')
        if row.get('metric') in ARCHIVED_METRICS:
            rows.append({
                'unit_id': candidate.candidate_id, 'kind': base['kind'],
                'batch': base['batch'], 'cell_id': base['cell_id'], 'split': base['split'],
                'p14_index': candidate.index,
                'k_BACE_multiplier': candidate.k_BACE_multiplier,
                'C_i': candidate.C_i, 'I_ext': candidate.I_ext,
                'arm': arm_name, 'intervention_kind': intervention.kind,
                'strength': intervention.strength, 'checkpoint': row.get('checkpoint'),
                'metric': row.get('metric'), 'value': row.get('value'),
                'reason': row.get('reason'), 'sample_time': row.get('sample_time'),
            })
    _require(len(rows) == 32 and {float(row['checkpoint']) for row in rows}
             == {200., 500., 1000., 2000.},
             f'{candidate.candidate_id}/{arm_name}: main outcomes lack 4x8 coverage.')
    return rows


def _contrast_rows(unit: JointSubject | CohortCandidate, arm_name: str,
                   intervention: object, treatment: list[dict],
                   control: list[dict] | None) -> list[dict]:
    candidate = _unit_candidate(unit)
    base = _unit_record(unit)
    rows = []
    for row in contrast_outcomes(treatment, control):
        if row['metric'] not in ARCHIVED_METRICS:
            continue
        rows.append({
            'unit_id': candidate.candidate_id, 'batch': base['batch'],
            'cell_id': base['cell_id'], 'split': base['split'],
            'p14_index': candidate.index,
            'k_BACE_multiplier': candidate.k_BACE_multiplier,
            'C_i': candidate.C_i, 'I_ext': candidate.I_ext,
            'arm': arm_name, 'intervention_kind': intervention.kind,
            'strength': intervention.strength, 'checkpoint': row['checkpoint'],
            'metric': row['metric'],
            'is_primary': row['metric'] == 'AUC_A42' and row['checkpoint'] == 2000.,
            'value': row['value'], 'control_value': row['control_value'],
            'difference_from_control': row['difference_from_control'],
            'percent_change_from_control': row['percent_change_from_control'],
            'contrast_reason': row['contrast_reason'],
        })
    _require(len(rows) == 32,
             f'{candidate.candidate_id}/{arm_name}: contrast lacks 4x8 coverage.')
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == fields,
                 f'{path}: unexpected columns.')
        return list(reader)


def _save_progress(path: Path, record: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _resume_completed(
    plan: JointPlan,
    batch: int,
    output: Path,
    units: tuple[JointSubject | CohortCandidate, ...],
    starting: dict[str, dict],
    gate_source: dict | None,
    source_hashes: dict[str, str],
    settings_hashes: dict[str, str],
) -> tuple[dict[str, dict], list[dict], list[dict], tuple[str, ...], tuple[str, ...]]:
    _require(output.parent == plan.output_root / f'batch_{batch:02d}' and output.is_dir(),
             '--resume-dir must be an existing attempt of the selected JT1 batch.')
    _require(not (output / 'summary.json').exists(),
             'This JT1 attempt already has a final summary and cannot be resumed.')
    inputs = _read_json(output / 'inputs.json')
    unit_raw = _unit_plan_raw(units)
    _require(sha256((output / 'interventions_joint_final.yaml').read_bytes())
                 == plan.config_hash
             and sha256((output / 'interventions.yaml').read_bytes()) == plan.catalog_hash
             and sha256((output / 'final.yaml').read_bytes()) == plan.final_plan.config_hash
             and sha256((output / 'p14_summary.json').read_bytes()) == plan.p14_summary_hash
             and sha256((output / 'p12_summary.json').read_bytes()) == plan.p12_summary_hash
             and (output / 'subject_allocation.csv').read_bytes() == plan.allocation_raw
             and (output / 'subjects.csv').read_bytes() == plan.subject_raw
             and (output / 'unit_plan.csv').read_bytes() == unit_raw,
             'Archived JT1 configuration, lineage or manifests changed.')
    _require(inputs.get('profile') == plan.profile
             and inputs.get('batch') == batch
             and inputs.get('batch_count') == plan.batch_count
             and inputs.get('config_sha256') == plan.config_hash
             and inputs.get('interventions_sha256') == plan.catalog_hash
             and inputs.get('p14_summary_sha256') == plan.p14_summary_hash
             and inputs.get('p12_summary_sha256') == plan.p12_summary_hash
             and inputs.get('allocation_manifest_sha256') == plan.allocation_hash
             and inputs.get('subject_manifest_sha256') == plan.subject_hash
             and inputs.get('unit_plan_sha256') == sha256(unit_raw)
             and inputs.get('unit_ids') == [_unit_id(unit) for unit in units]
             and inputs.get('starting_states') == starting
             and inputs.get('joint_gate_source') == gate_source
             and inputs.get('source_sha256') == source_hashes
             and inputs.get('settings_sha256') == settings_hashes,
             'Resume lineage, assignments, starting states or source differ.')
    for name, digest in source_hashes.items():
        _require(sha256((output / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'Archived JT1 implementation changed: {name}.')

    reports: dict[str, dict] = {}
    outcome_rows: list[dict] = []
    contrast_rows: list[dict] = []
    gap = False
    partial_ids: list[str] = []
    for unit in units:
        unit_id = _unit_id(unit)
        case_dir = output / 'cases' / unit_id
        decision_path = case_dir / 'decision.json'
        if not decision_path.exists():
            if case_dir.exists():
                _require(not gap,
                         f'{unit_id}: files occur after an incomplete prefix gap.')
                partial_ids.append(unit_id)
            gap = True
            continue
        _require(not gap, 'Completed JT1 units must form a contiguous gate/subject prefix.')
        report = _read_json(decision_path)
        _require(report.get('unit') == _unit_record(unit)
                 and report.get('starting_state') == starting[unit_id]
                 and report.get('integrations_attempted') == len(ARMS) * len(VARIANTS)
                 and isinstance(report.get('numerically_confirmed'), bool)
                 and isinstance(report.get('issues'), list),
                 f'{unit_id}: archived JT1 decision differs from this design.')
        comparisons = mapping(report.get('comparisons'), f'{unit_id} comparisons')
        _require(set(comparisons) == set(ARMS), f'{unit_id}: arm comparisons are incomplete.')
        for arm_name in ARMS:
            checks = mapping(comparisons[arm_name], f'{unit_id}/{arm_name} comparisons')
            _require(set(checks) == {label for label, _, _ in COMPARISONS},
                     f'{unit_id}/{arm_name}: comparison coverage changed.')
            for variant in VARIANTS:
                run_dir = case_dir / arm_name / variant
                run = _read_json(run_dir / 'run.json')
                _require(run.get('status') == report['arms'][arm_name][variant]['status']
                         and run.get('settings_sha256') == settings_hashes[variant]
                         and (run_dir / 'outcomes.json').is_file(),
                         f'{unit_id}/{arm_name}/{variant}: archived run differs.')
                if run.get('status') == 'completed':
                    _require(run.get('trajectory_file') == 'trajectory.npz'
                             and (run_dir / 'trajectory.npz').is_file(),
                             f'{unit_id}/{arm_name}/{variant}: trajectory is missing.')
        unit_outcomes = _read_csv(case_dir / 'outcomes.csv', OUTCOME_FIELDS)
        unit_contrasts = _read_csv(case_dir / 'paired_contrasts.csv', CONTRAST_FIELDS)
        if report.get('numerically_confirmed') is True:
            _require(len(unit_outcomes) == len(ARMS) * 32
                     and len(unit_contrasts) == (len(ARMS) - 1) * 32,
                     f'{unit_id}: confirmed unit lost outcome or contrast coverage.')
        reports[unit_id] = report
        if isinstance(unit, JointSubject):
            outcome_rows.extend(unit_outcomes)
            contrast_rows.extend(unit_contrasts)
    aggregate_files: list[str] = []
    for name, fields, rows in (
        ('outcomes.csv', OUTCOME_FIELDS, outcome_rows),
        ('paired_contrasts.csv', CONTRAST_FIELDS, contrast_rows),
    ):
        path = output / name
        if path.exists():
            _require(_read_csv(path, fields) == rows,
                     f'Existing partial-finalization artifact differs: {name}.')
            aggregate_files.append(name)
    return (reports, outcome_rows, contrast_rows, tuple(partial_ids),
            tuple(aggregate_files))


def _execute(
    plan: JointPlan,
    batch: int,
    subjects: tuple[JointSubject, ...],
    gate_source: dict | None,
    starting: dict[str, dict],
    interval: float,
    show_progress: bool,
    resume_dir: Path | None,
) -> int:
    sources = _source_files()
    source_hashes = {name: sha256(raw) for name, raw in sources.items()}
    settings_hashes = {
        name: _canonical_hash(asdict(value))
        for name, value in plan.final_plan.settings.items()
    }
    units: tuple[JointSubject | CohortCandidate, ...] = (
        (*plan.final_plan.gate, *subjects) if batch == 1 else subjects
    )
    unit_raw = _unit_plan_raw(units)
    if resume_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = plan.output_root / f'batch_{batch:02d}' / stamp
        output.mkdir(parents=True, exist_ok=False)
        (output / 'interventions_joint_final.yaml').write_bytes(plan.raw)
        (output / 'interventions.yaml').write_bytes(plan.catalog_raw)
        (output / 'final.yaml').write_bytes(plan.final_config_raw)
        (output / 'p14_summary.json').write_bytes(plan.p14_summary_raw)
        (output / 'p12_summary.json').write_bytes(plan.p12_summary_raw)
        (output / 'subject_allocation.csv').write_bytes(plan.allocation_raw)
        (output / 'subjects.csv').write_bytes(plan.subject_raw)
        (output / 'unit_plan.csv').write_bytes(unit_raw)
        for name, raw in sources.items():
            copy = output / 'implementation' / 'adt' / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(raw)
        reports: dict[str, dict] = {}
        outcome_rows: list[dict] = []
        contrast_rows: list[dict] = []
        _json(output / 'inputs.json', {
            'profile': plan.profile, 'model_version': '0.3',
            'batch': batch, 'batch_count': plan.batch_count,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'config_sha256': plan.config_hash,
            'interventions_sha256': plan.catalog_hash,
            'p14_collection': _relative(plan.p14_dir),
            'p14_summary_sha256': plan.p14_summary_hash,
            'p14_candidates_sha256': plan.p14_candidates_hash,
            'p14_outcomes_sha256': plan.p14_outcomes_hash,
            'p12_collection': _relative(plan.p12_dir),
            'p12_summary_sha256': plan.p12_summary_hash,
            'p12_contrasts_sha256': plan.p12_contrasts_hash,
            'p12_descriptive_sha256': plan.p12_descriptive_hash,
            'allocation_manifest_sha256': plan.allocation_hash,
            'subject_manifest_sha256': plan.subject_hash,
            'derived_seeds': plan.seeds,
            'seed_derivation': 'sha256_utf8_first_128_bits_big_endian_to_numpy_pcg64',
            'unit_plan_sha256': sha256(unit_raw),
            'unit_ids': [_unit_id(unit) for unit in units],
            'subject_ids': [row.candidate.candidate_id for row in subjects],
            'joint_gate_ids': ([row.candidate_id for row in plan.final_plan.gate]
                               if batch == 1 else []),
            'joint_gate_source': gate_source,
            'starting_states': starting,
            'state_order': STATE_NAMES,
            'arms': {name: asdict(plan.arms[name]) for name in ARMS},
            'variants': VARIANTS,
            'settings_sha256': settings_hashes,
            'checkpoints': plan.checkpoints,
            'primary_metric': 'AUC_A42', 'primary_checkpoint': 2000.,
            'responder_classification': 'continuous_only',
            'trajectory_archive': 'npz_compressed_all_variants',
            'source_sha256': source_hashes,
            'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'scipy', 'PyYAML')},
        })
    else:
        output = resume_dir.resolve()
        reports, outcome_rows, contrast_rows, partial_ids, aggregate_files = _resume_completed(
            plan, batch, output, units, starting, gate_source,
            source_hashes, settings_hashes)
        if partial_ids or aggregate_files:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            archived: dict[str, str] = {}
            for unit_id in partial_ids:
                source = output / 'cases' / unit_id
                destination = output / 'incomplete_units' / f'{unit_id}_{stamp}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                archived[unit_id] = str(destination.relative_to(output).as_posix())
            archived_aggregates: dict[str, str] = {}
            for name in aggregate_files:
                source = output / name
                destination = output / 'incomplete_aggregates' / f'{stamp}_{name}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                archived_aggregates[name] = str(destination.relative_to(output).as_posix())
            _json(output / f'resumption_{stamp}.json', {
                'resumed_utc': datetime.now(timezone.utc).isoformat(),
                'complete_units_reused': list(reports),
                'partial_units_archived': archived,
                'partial_finalization_artifacts_archived': archived_aggregates,
                'policy': 'archive_partial_state_then_repeat_unit_or_rebuild_aggregates',
            })

    total = len(units) * len(ARMS) * len(VARIANTS)
    progress = None if not show_progress else ProgressBar(
        total, label=f'P15 JT1 lote {batch}/{plan.batch_count}',
        unit='integraciones', interval=interval)
    if progress is not None:
        progress.done = len(reports) * len(ARMS) * len(VARIANTS)
        progress.start()

    def run_one(unit: JointSubject | CohortCandidate) -> None:
        unit_id = _unit_id(unit)
        if unit_id in reports:
            return
        candidate = _unit_candidate(unit)
        case = _candidate_case(plan.final_plan, candidate)
        initial = state_vector([starting[unit_id]['state'][name] for name in STATE_NAMES])
        runs_by_arm: dict[str, dict[str, SweepRun]] = {}
        comparisons: dict[str, dict] = {}
        issues: list[str] = []
        unit_outcomes: list[dict] = []
        for arm_name in ARMS:
            intervention = plan.arms[arm_name]
            runs: dict[str, SweepRun] = {}
            for variant in VARIANTS:
                phase = f'{unit_id}/{arm_name}/{variant}'
                if progress is not None:
                    progress.stage(f'{phase}: integrando')
                run = _run_arm(case.parameters, case.subject, case.external_input,
                               intervention, initial, plan.final_plan.settings[variant],
                               plan.checkpoints)
                runs[variant] = run
                _save_run(output, unit_id, arm_name, variant, run,
                          settings_hashes[variant])
                if progress is not None:
                    progress.advance(f'{phase}: {run.status}; archivo guardado')
                if run.status != 'completed':
                    issues.append(f'{arm_name}/{variant}: {run.message}')
            runs_by_arm[arm_name] = runs
            comparisons[arm_name] = {
                label: _compare(runs[left], runs[right], plan.final_plan.agreement)
                for label, left, right in COMPARISONS
            }
            issues.extend(
                f'{arm_name}/{label}: numerical_agreement_failed'
                for label, result in comparisons[arm_name].items()
                if result.get('passes') is not True
            )
            if runs['main'].status == 'completed':
                rows = _outcome_rows(unit, arm_name, intervention, runs['main'].outcomes)
                unit_outcomes.extend(rows)
                issues.extend(
                    f'{arm_name}/{row["metric"]}@{row["checkpoint"]}: {row["reason"]}'
                    for row in rows if row['value'] is None
                    and not (row['metric'] == 'Ratio_42_40'
                             and row['reason'] == 'A40_zero_at_checkpoint')
                )
        control = (runs_by_arm['control']['main'].outcomes
                   if runs_by_arm['control']['main'].status == 'completed' else None)
        unit_contrasts: list[dict] = []
        for arm_name in ARMS[1:]:
            treatment = runs_by_arm[arm_name]['main']
            if treatment.status == 'completed':
                unit_contrasts.extend(_contrast_rows(
                    unit, arm_name, plan.arms[arm_name], treatment.outcomes, control))
        issues.extend(
            f'{row["arm"]}/AUC_A42@2000: {row["contrast_reason"]}'
            for row in unit_contrasts
            if row['is_primary'] and row['contrast_reason'] not in (None, '')
        )
        confirmed = not issues
        report = {
            'unit': _unit_record(unit),
            'parameters': asdict(case.parameters),
            'subject': asdict(case.subject),
            'external_input': asdict(case.external_input),
            'starting_state': starting[unit_id],
            'arms': {
                arm_name: {
                    variant: {'status': run.status, 'message': run.message}
                    for variant, run in runs.items()
                } for arm_name, runs in runs_by_arm.items()
            },
            'comparisons': comparisons,
            'integrations_attempted': len(ARMS) * len(VARIANTS),
            'numerically_confirmed': confirmed,
            'accepted': confirmed,
            'technical_acceptance_only': True,
            'biological_status': 'unclassified',
            'responder_classification': 'not_assessed',
            'issues': issues,
        }
        case_dir = output / 'cases' / unit_id
        _write_csv(case_dir / 'outcomes.csv', OUTCOME_FIELDS, unit_outcomes)
        _write_csv(case_dir / 'paired_contrasts.csv', CONTRAST_FIELDS, unit_contrasts)
        _json(case_dir / 'decision.json', report)
        reports[unit_id] = report
        if isinstance(unit, JointSubject):
            outcome_rows.extend(unit_outcomes)
            contrast_rows.extend(unit_contrasts)
        subject_reports = [reports[row.candidate.candidate_id] for row in subjects
                           if row.candidate.candidate_id in reports]
        gate_reports = [reports[row.candidate_id] for row in plan.final_plan.gate
                        if row.candidate_id in reports]
        _save_progress(output / 'progress.json', {
            'last_unit': unit_id,
            'joint_gate_attempted': len(gate_reports),
            'joint_gate_confirmed': sum(row['numerically_confirmed'] for row in gate_reports),
            'subjects_attempted': len(subject_reports),
            'subjects_numerically_confirmed': sum(
                row['numerically_confirmed'] for row in subject_reports),
            'integrations_attempted': len(reports) * len(ARMS) * len(VARIANTS),
        })

    try:
        if batch == 1:
            for gate in plan.final_plan.gate:
                run_one(gate)
            gate_ok = all(reports[row.candidate_id]['numerically_confirmed']
                          for row in plan.final_plan.gate)
            if gate_ok:
                for subject in subjects:
                    run_one(subject)
        else:
            gate_ok = True
            for subject in subjects:
                run_one(subject)
    finally:
        if progress is not None:
            progress.close()

    _write_csv(output / 'outcomes.csv', OUTCOME_FIELDS, outcome_rows)
    _write_csv(output / 'paired_contrasts.csv', CONTRAST_FIELDS, contrast_rows)
    subject_reports = [reports[row.candidate.candidate_id] for row in subjects
                       if row.candidate.candidate_id in reports]
    confirmed_subjects = sum(row['numerically_confirmed'] for row in subject_reports)
    complete = len(subject_reports) == len(subjects)
    confirmed = gate_ok and complete and confirmed_subjects == len(subjects)
    status = 'confirmed' if confirmed else ('gate_not_confirmed' if not gate_ok
                                             else 'not_confirmed')
    gate_reports = [reports[row.candidate_id] for row in plan.final_plan.gate
                    if row.candidate_id in reports]
    checks_passing = sum(
        check.get('passes') is True
        for report in reports.values()
        for arm_checks in report['comparisons'].values()
        for check in arm_checks.values()
    )
    summary = {
        'profile': plan.profile, 'status': status,
        'numerically_confirmed': confirmed,
        'technical_acceptance_only': True,
        'biological_status': 'unclassified',
        'population_inference': False,
        'responder_classification': 'not_assessed',
        'batch': batch, 'batch_count': plan.batch_count,
        'batch_size': plan.batch_size,
        'subjects_planned': len(subjects),
        'subjects_attempted': len(subject_reports),
        'subjects_numerically_confirmed': confirmed_subjects,
        'subjects_rejected': sum(not row['numerically_confirmed'] for row in subject_reports),
        'joint_gate_required': True,
        'joint_gate_local_points': len(plan.final_plan.gate) if batch == 1 else 0,
        'joint_gate_points_confirmed': sum(
            row['numerically_confirmed'] for row in gate_reports),
        'joint_gate_confirmed': gate_ok,
        'joint_gate_source': gate_source,
        'arms_per_subject': len(ARMS), 'variants_per_arm': len(VARIANTS),
        'integrations_planned': total,
        'integrations_attempted': len(reports) * len(ARMS) * len(VARIANTS),
        'comparison_checks_planned': len(units) * len(ARMS) * len(COMPARISONS),
        'comparison_checks_passing': checks_passing,
        'outcome_rows': len(outcome_rows),
        'contrast_rows': len(contrast_rows),
        'config_sha256': plan.config_hash,
        'interventions_sha256': plan.catalog_hash,
        'p14_summary_sha256': plan.p14_summary_hash,
        'p12_summary_sha256': plan.p12_summary_hash,
        'allocation_manifest_sha256': plan.allocation_hash,
        'subject_manifest_sha256': plan.subject_hash,
        'unit_plan_sha256': sha256(unit_raw),
        'source_sha256': source_hashes,
        'settings_sha256': settings_hashes,
        'issues': [f'{unit_id}: {issue}' for unit_id, report in reports.items()
                   for issue in report['issues']],
        'outcomes_file': 'outcomes.csv',
        'contrasts_file': 'paired_contrasts.csv',
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    print(f'\n{status}: {confirmed_subjects}/{len(subjects)} sujetos; '
          f'{summary["integrations_attempted"]}/{total} integraciones intentadas')
    print(f'Artifacts: {output}')
    return 0 if confirmed else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run one selected 128-subject P15 JT1-A2FULL batch.')
    parser.add_argument('--config', type=Path,
                        default=Path('config/interventions_joint_final.yaml'))
    parser.add_argument('--batch', type=int, required=True, help='Batch number, 1 through 8.')
    parser.add_argument('--gate-dir', type=Path,
                        help='Confirmed batch-1 directory when automatic selection is ambiguous.')
    parser.add_argument('--check-only', action='store_true',
                        help='Read-only preflight without RHS, integration or writes.')
    parser.add_argument('--resume-dir', type=Path,
                        help='Continue a contiguous prefix of complete JT1 units.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = _load_plan(args.config)
        subjects = plan.select(args.batch)
        _require(0 < args.progress_interval < float('inf'),
                 '--progress-interval must be positive and finite.')
        sources = {name: sha256(raw) for name, raw in _source_files().items()}
        settings = {name: _canonical_hash(asdict(value))
                    for name, value in plan.final_plan.settings.items()}
        gate_source = _gate_source(plan, args.batch, args.gate_dir, sources, settings)
        units: tuple[JointSubject | CohortCandidate, ...] = (
            (*plan.final_plan.gate, *subjects) if args.batch == 1 else subjects
        )
        starting = _starting_states(plan, units)
        reused = 0
        partial = 0
        partial_aggregates = 0
        if args.resume_dir is not None:
            recovered = _resume_completed(
                plan, args.batch, args.resume_dir.resolve(), units, starting,
                gate_source, sources, settings)
            reused = len(recovered[0])
            partial = len(recovered[3])
            partial_aggregates = len(recovered[4])
    except (OSError, UnicodeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError, yaml.YAMLError) as error:
        parser.error(str(error))
    total = len(units) * len(ARMS) * len(VARIANTS)
    if args.check_only:
        print(f'P15 JT1 preflight OK: lote {args.batch}/{plan.batch_count}, '
              f'{len(subjects)} sujetos, {total} integraciones; '
              f'{reused * len(ARMS) * len(VARIANTS)} reutilizables.')
        if partial:
            print(f'{partial} unidad parcial se archivará y se repetirá completa al reanudar.')
        if partial_aggregates:
            print(f'{partial_aggregates} agregado parcial se archivará y se reconstruirá.')
        print(f'Manifiestos: asignación {plan.allocation_hash}; sujetos {plan.subject_hash}.')
        if args.batch == 1:
            print('La puerta P14 de 8 esquinas + centro se ejecutará antes de los sujetos.')
        else:
            print(f'Puerta JT1 fijada por {gate_source["directory"]}.')
        print('Sin RHS, integración ni archivos nuevos.')
        return 0
    return _execute(plan, args.batch, subjects, gate_source, starting,
                    args.progress_interval, not args.no_progress, args.resume_dir)
