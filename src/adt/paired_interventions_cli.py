"""Manual stage-12 paired interventions in independent 20-subject batches."""

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from .basal_cli import _json
from .inputs import ConstantInput
from .interventions import Intervention, intervention_from_config
from .model import state_vector
from .outcomes import (
    AgreementSettings, at_times, compare_outcomes, compare_trajectories,
    contrast_outcomes, evaluate_outcomes, validate_trajectory,
)
from .parameters import ConfigDraft, ModelParameters, SubjectParameters, finite_number, load_config, mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .simulate import SimulationError, SimulationSettings, simulate
from .states import STATE_NAMES
from .sweep_cli import _save_run
from .sweeps import SweepRun


VARIANTS = ('main', 'radau_refined', 'bdf_refined')
GROUPS = ('production_driven', 'clearance_deficient', 'inflammation_driven')
GROUP_PREFIXES = {'production_driven': 'k_BACE_',
                  'clearance_deficient': 'C_i_',
                  'inflammation_driven': 'I_ext_'}
COMPARISONS = (('main_vs_radau_refined', 'main', 'radau_refined'),
               ('refined_methods', 'radau_refined', 'bdf_refined'))
SUPPLEMENTARY = ('A42_final', 'Ratio_42_40', 'AUC_O', 'P_final',
                 'AUC_I', 'C99_max', 'AUC_C99')


@dataclass(frozen=True)
class PairedPlan:
    profile: str
    config_hash: str
    catalog_hash: str
    pilot_dir: Path
    pilot_summary_hash: str
    pilot_inputs_hash: str
    output_root: Path
    batch_size: int
    checkpoints: tuple[float, ...]
    primary_checkpoint: float
    settings: dict[str, SimulationSettings]
    agreement: AgreementSettings
    arms: dict[str, Intervention]
    groups: dict[str, tuple[str, ...]]
    cases: dict[str, dict]

    @property
    def batches(self) -> int:
        return sum(len(ids) // self.batch_size for ids in self.groups.values())

    def select(self, number: int) -> tuple[str, tuple[str, ...]]:
        if type(number) is not int or not 1 <= number <= self.batches:
            raise ValueError(f'Select a batch between 1 and {self.batches}.')
        index = number - 1
        for group in GROUPS:
            count = len(self.groups[group]) // self.batch_size
            if index < count:
                start = index * self.batch_size
                return group, self.groups[group][start:start + self.batch_size]
            index -= count
        raise ValueError('Batch selection failed.')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    return mapping(value, str(path))


def _read_pinned(path: Path, expected: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    _require(sha256(raw) == expected, f'SHA-256 mismatch: {path}')
    return raw, mapping(json.loads(raw), str(path))


def _config(path: Path) -> tuple[dict, bytes, str]:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _require(set(data) == {'profile', 'status', 'source', 'design', 'evaluation',
                           'outcomes', 'output_root'}, 'Unexpected stage-12 config fields.')
    _require(data['profile'] == 'P12_C1_paired' and data['status'] == 'selected',
             'Stage-12 profile must be selected.')
    source = mapping(data['source'], 'source')
    _require(set(source) == {'pilot_dir', 'pilot_summary_sha256',
                             'interventions_file', 'interventions_sha256'},
             'Incomplete or unsupported source section.')
    design = mapping(data['design'], 'design')
    _require(design == {'initial_state': 'c1_main_final_state_at_2000',
                        'paired_control': 'own_subject_control', 'batch_size': 20,
                        'group_order': list(GROUPS), 'numerical_variants': list(VARIANTS)},
             'Stage-12 design differs from the selected 20-subject paired protocol.')
    evaluation = mapping(data['evaluation'], 'evaluation')
    _require(evaluation == {'horizon': 2000.0,
                            'checkpoints': [200.0, 500.0, 1000.0, 2000.0]},
             'Stage-12 horizon/checkpoints must match the selected scheme.')
    outcomes = mapping(data['outcomes'], 'outcomes')
    _require(outcomes == {'primary_metric': 'AUC_A42',
                          'primary_checkpoint': 2000.0,
                          'contrast': 'percent_change_from_own_control',
                          'supplementary_metrics': list(SUPPLEMENTARY),
                          'responder_classification': 'continuous_only',
                          'biological_status': 'unclassified'},
             'Stage-12 outcomes differ from the selected continuous analysis.')
    _require(data['output_root'] == 'results/interventions/paired',
             'Stage-12 output must remain under results/interventions/paired.')
    return data, raw, sha256(raw)


def _arm_catalog(path: Path, expected_hash: str) -> tuple[dict[str, Intervention], bytes]:
    raw = path.read_bytes()
    _require(sha256(raw) == expected_hash, 'Intervention catalog hash changed.')
    catalog = ConfigDraft(path, mapping(yaml.safe_load(raw.decode('utf-8')), 'interventions'))
    gsi = mapping(catalog.data['GSI'], 'GSI')['rho_GSI_levels']
    _require(isinstance(gsi, list) and len(gsi) == 4,
             'Four fixed v0.3 GSI levels are required.')
    strengths = tuple(finite_number(level, 'GSI level') for level in gsi)
    _require(all(0 < x < 1 for x in strengths)
             and all(a < b for a, b in zip(strengths, strengths[1:])),
             'GSI levels must be increasing strengths in (0,1).')
    arms = {'control': intervention_from_config(catalog, 'control')}
    for index in range(4):
        arms[f'GSI_{index + 1:02d}'] = intervention_from_config(catalog, 'GSI', level=index)
    for label in ('low', 'medium', 'high'):
        arms[f'GSM_{label}'] = intervention_from_config(catalog, 'GSM', level=label)
    gsm = tuple(arms[f'GSM_{label}'].strength for label in ('low', 'medium', 'high'))
    _require(all(0 < x < 1 for x in gsm) and gsm[0] < gsm[1] < gsm[2],
             'GSM levels must be increasing strengths in (0,1).')
    return arms, raw


def _case_groups(inputs: dict) -> dict[str, tuple[str, ...]]:
    cases = mapping(inputs['cases'], 'C1 cases')
    _require(len(cases) == 121 and 'baseline' in cases, 'C1 must contain 120 subjects and one basal.')
    expected = {f'{prefix}{index:03d}' for prefix in GROUP_PREFIXES.values()
                for index in range(1, 41)}
    _require(set(cases) == expected | {'baseline'}, 'C1 subject IDs do not match the 40/40/40 design.')
    return {group: tuple(f'{prefix}{index:03d}' for index in range(1, 41))
            for group, prefix in GROUP_PREFIXES.items()}


def _source_settings(inputs: dict, checkpoints: tuple[float, ...]) -> tuple[dict, AgreementSettings]:
    archived = mapping(inputs['settings'], 'C1 settings')
    _require(set(archived) == set(VARIANTS), 'C1 solver variants changed.')
    settings = {name: SimulationSettings(**mapping(archived[name], name)) for name in VARIANTS}
    main = settings['main']
    _require(main.t_span == (0., 2000.)
             and all(value in main.t_eval for value in checkpoints),
             'C1 main grid does not cover the selected treatment checkpoints.')
    for name in VARIANTS[1:]:
        other = settings[name]
        _require(other.t_span == main.t_span and set(main.t_eval) <= set(other.t_eval),
                 f'C1 {name} grid does not contain all main samples.')
    _require(main.method == 'Radau' and settings['radau_refined'].method == 'Radau'
             and settings['bdf_refined'].method == 'BDF', 'C1 solver methods changed.')
    return settings, AgreementSettings(**mapping(inputs['agreement'], 'C1 agreement'))


def _preflight(path: Path, batch: int) -> tuple[PairedPlan, tuple[str, ...], bytes, bytes, dict]:
    data, config_raw, config_hash = _config(path)
    source = data['source']
    pilot_dir = Path(source['pilot_dir']).resolve()
    summary_raw, summary = _read_pinned(pilot_dir / 'summary.json',
                                        source['pilot_summary_sha256'])
    _require(summary.get('status') == 'confirmed'
             and summary.get('numerically_confirmed') is True
             and summary.get('candidates_accepted') == 120
             and summary.get('candidates_rejected') == 0
             and summary.get('issues') == [], 'Pinned C1 pilot is not fully confirmed.')
    _, inputs = _read_pinned(pilot_dir / 'inputs.json', summary['inputs_sha256'])
    _require(sha256((pilot_dir / 'pilot.yaml').read_bytes()) == summary['config_sha256']
             and inputs.get('config_sha256') == summary['config_sha256']
             and inputs.get('state_order') == list(STATE_NAMES),
             'C1 configuration or state order changed.')
    groups = _case_groups(inputs)
    settings, agreement = _source_settings(inputs, (200., 500., 1000., 2000.))
    arms, catalog_raw = _arm_catalog(Path(source['interventions_file']).resolve(),
                                     source['interventions_sha256'])
    output_root = Path(data['output_root']).resolve()
    results_root = Path('results').resolve()
    _require(output_root != results_root and output_root.is_relative_to(results_root),
             'Stage-12 output must be a child of results/.')
    plan = PairedPlan(data['profile'], config_hash, sha256(catalog_raw), pilot_dir,
                      sha256(summary_raw), summary['inputs_sha256'], output_root,
                      data['design']['batch_size'], (200., 500., 1000., 2000.),
                      data['outcomes']['primary_checkpoint'],
                      settings, agreement, arms, groups, inputs['cases'])
    group, ids = plan.select(batch)
    _require(len(ids) == 20, 'Invalid batch selection.')
    starting = {}
    for case_id in ids:
        case = mapping(plan.cases[case_id], case_id)
        decision = _read_json(pilot_dir / 'cases' / case_id / 'decision.json')
        _require(decision.get('case_id') == case_id and decision.get('phenotype') == group
                 and decision.get('accepted') is True
                 and decision.get('parameters') == case.get('parameters')
                 and decision.get('subject') == case.get('subject')
                 and decision.get('external_input') == case.get('external_input'),
                 f'{case_id}: C1 accepted case mismatch.')
        run_path = pilot_dir / 'cases' / case_id / 'main' / 'run.json'
        run_raw = run_path.read_bytes()
        run = mapping(json.loads(run_raw), str(run_path))
        _require(run.get('status') == 'completed' and run.get('final_time') == 2000.,
                 f'{case_id}: C1 main run lacks its completed 2000-tau endpoint.')
        initial = mapping(run.get('final_state'), f'{case_id}.final_state')
        _require(set(initial) == set(STATE_NAMES), f'{case_id}: incomplete C1 final state.')
        state_vector([finite_number(initial[name], f'{case_id}.{name}') for name in STATE_NAMES])
        metadata = mapping(run.get('metadata'), f'{case_id}.metadata')
        _require(metadata.get('parameters') == case['parameters']
                 and metadata.get('subject') == case['subject']
                 and metadata.get('input') == {'kind': 'ConstantInput', **case['external_input']}
                 and metadata.get('intervention') == {'kind': 'control', 'strength': None},
                 f'{case_id}: C1 run metadata differs from its subject.')
        starting[case_id] = {'state': initial, 'run_sha256': sha256(run_raw)}
    return plan, ids, config_raw, catalog_raw, starting


def _run_arm(parameters: ModelParameters, subject: SubjectParameters,
             external: ConstantInput, intervention: Intervention,
             initial: np.ndarray, settings: SimulationSettings,
             checkpoints: tuple[float, ...]) -> SweepRun:
    result = None
    try:
        result = simulate(parameters, subject, external, intervention,
                          initial.copy(), settings)
        times, states = validate_trajectory(result.t, result.y)
        if (not result.success or result.state_names != STATE_NAMES
                or not np.array_equal(times, settings.t_eval)
                or not np.array_equal(states[:, 0], initial)):
            raise SimulationError('Treatment run lacks the requested complete grid or starting state.')
        outcomes = evaluate_outcomes(times, states, checkpoints)
        return SweepRun('completed', result.message, settings, result, outcomes)
    except (SimulationError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        return SweepRun('failed', f'{type(error).__name__}: {error}', settings, result, [])


def _compare(left: SweepRun, right: SweepRun, agreement: AgreementSettings) -> dict:
    if left.status != 'completed' or right.status != 'completed':
        return {'passes': False, 'reason': 'run_unavailable'}
    try:
        trajectory = compare_trajectories(left.result.y,
            at_times(right.result.t, right.result.y, left.result.t), agreement)
        outcomes = compare_outcomes(left.outcomes, right.outcomes, agreement)
        return {'passes': trajectory['passes'] and outcomes['passes'],
                'trajectory': trajectory, 'outcomes': outcomes}
    except (ValueError, ArithmeticError) as error:
        return {'passes': False, 'reason': str(error)}


def _write_contrasts(path: Path, rows: list[dict]) -> None:
    columns = ('case_id', 'phenotype', 'arm', 'kind', 'strength', 'checkpoint',
               'metric', 'is_primary', 'value', 'control_value',
               'difference_from_control', 'percent_change_from_control', 'contrast_reason')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


def _save_progress(path: Path, record: dict) -> None:
    """Atomically replace the mutable progress checkpoint after each subject."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _resume_completed(plan: PairedPlan, ids: tuple[str, ...], batch: int,
                      output: Path, starting: dict,
                      current_hashes: dict[str, str]) -> tuple[dict, list[dict], dict | None]:
    expected_parent = plan.output_root / f'batch_{batch:02d}'
    _require(output.parent == expected_parent and output.is_dir(),
             '--resume-dir must be an existing run of the selected batch.')
    _require(not (output / 'summary.json').exists()
             and not (output / 'paired_contrasts.csv').exists(),
             'This batch already has a final or aggregate artifact; it cannot be resumed.')
    archived = _read_json(output / 'inputs.json')
    _require(sha256((output / 'interventions_paired.yaml').read_bytes()) == plan.config_hash
             and sha256((output / 'interventions.yaml').read_bytes()) == plan.catalog_hash,
             'Archived batch configuration or intervention catalog changed.')
    group, _ = plan.select(batch)
    _require(archived.get('profile') == plan.profile
             and archived.get('batch') == batch
             and archived.get('batch_count') == plan.batches
             and archived.get('phenotype') == group
             and archived.get('case_ids') == list(ids)
             and archived.get('config_sha256') == plan.config_hash
             and archived.get('interventions_sha256') == plan.catalog_hash
             and archived.get('pilot_summary_sha256') == plan.pilot_summary_hash
             and archived.get('pilot_inputs_sha256') == plan.pilot_inputs_hash
             and archived.get('starting_states') == starting,
             'Resume source, subject assignments or starting states differ.')
    old_hashes = mapping(archived.get('source_sha256'), 'archived source hashes')
    for name, digest in old_hashes.items():
        _require(sha256((output / 'implementation' / 'adt' / name).read_bytes()) == digest,
                 f'Archived implementation snapshot changed: {name}.')
    changed = sorted(name for name in old_hashes.keys() | current_hashes.keys()
                     if old_hashes.get(name) != current_hashes.get(name))
    _require(set(changed) <= {'paired_interventions_cli.py', 'paired_collection_cli.py'},
             'Scientific/model source changed; do not mix implementations in one batch.')
    resumption_path = output / 'resumption.json'
    resumption = _read_json(resumption_path) if resumption_path.exists() else None
    if resumption is not None:
        _require(resumption.get('previous_source_sha256') == old_hashes
                 and resumption.get('effective_source_sha256') == current_hashes
                 and resumption.get('changed_files') == changed,
                 'Existing resumption record differs from the current source.')
        for name, digest in current_hashes.items():
            _require(sha256((output / 'resumption_implementation' / 'adt' / name).read_bytes())
                     == digest, f'Resumption source snapshot changed: {name}.')
    reports: dict[str, dict] = {}
    rows: list[dict] = []
    found_gap = False
    expected_arms = set(plan.arms)
    for case_id in ids:
        case_dir = output / 'cases' / case_id
        decision_path = case_dir / 'decision.json'
        if not decision_path.exists():
            _require(not case_dir.exists(),
                     f'{case_id}: partial subject files exist; archive them before a new attempt.')
            found_gap = True
            continue
        _require(not found_gap, 'Completed subjects must form a contiguous batch prefix.')
        report = _read_json(decision_path)
        _require(report.get('case_id') == case_id
                 and report.get('phenotype') == group
                 and report.get('c1_main_run_sha256') == starting[case_id]['run_sha256']
                 and report.get('initial_state') == starting[case_id]['state']
                 and set(mapping(report.get('arms'), 'archived arms')) == expected_arms
                 and isinstance(report.get('numerically_confirmed'), bool)
                 and isinstance(report.get('issues'), list),
                 f'{case_id}: archived subject decision differs from this design.')
        for arm_name, intervention in plan.arms.items():
            variants = mapping(report['arms'][arm_name], f'{case_id}/{arm_name}')
            _require(set(variants) == set(VARIANTS),
                     f'{case_id}/{arm_name}: archived solver variants are incomplete.')
            for variant in VARIANTS:
                run_dir = case_dir / arm_name / variant
                run = _read_json(run_dir / 'run.json')
                _require(run.get('status') == variants[variant]['status']
                         and (run_dir / 'outcomes.json').is_file(),
                         f'{case_id}/{arm_name}/{variant}: archived run differs from decision.')
                if run['status'] == 'completed':
                    metadata = mapping(run.get('metadata'), 'archived run metadata')
                    _require(run.get('final_time') == 2000.
                             and (run_dir / 'trajectory.csv').is_file()
                             and metadata.get('initial_state') ==
                                 [starting[case_id]['state'][name] for name in STATE_NAMES]
                             and metadata.get('intervention') == asdict(intervention),
                             f'{case_id}/{arm_name}/{variant}: archived run lineage differs.')
        with (case_dir / 'paired_contrasts.csv').open(newline='', encoding='utf-8') as stream:
            case_rows = list(csv.DictReader(stream))
        if report['numerically_confirmed']:
            _require(len(case_rows) == 7 * len(plan.checkpoints) * (1 + len(SUPPLEMENTARY)),
                     f'{case_id}: confirmed subject lacks complete paired contrasts.')
        _require(all(row.get('case_id') == case_id for row in case_rows),
                 f'{case_id}: archived paired contrasts refer to another subject.')
        reports[case_id] = report
        rows.extend(case_rows)
    return reports, rows, ({'previous_source_sha256': old_hashes,
                            'effective_source_sha256': current_hashes,
                            'changed_files': changed} if resumption is None else None)


def _execute(plan: PairedPlan, ids: tuple[str, ...], batch: int,
             config_raw: bytes, catalog_raw: bytes, starting: dict,
             interval: float, show_progress: bool,
             resume_dir: Path | None = None) -> int:
    sources = _source_files()
    source_hashes = {name: sha256(raw) for name, raw in sources.items()}
    if resume_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = plan.output_root / f'batch_{batch:02d}' / stamp
        output.mkdir(parents=True, exist_ok=False)
        (output / 'interventions_paired.yaml').write_bytes(config_raw)
        (output / 'interventions.yaml').write_bytes(catalog_raw)
        for name, raw in sources.items():
            copy = output / 'implementation' / 'adt' / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(raw)
        reports, rows, resumption = {}, [], None
    else:
        output = resume_dir.resolve()
        reports, rows, resumption = _resume_completed(
            plan, ids, batch, output, starting, source_hashes)
        if resumption is not None:
            for name, raw in sources.items():
                copy = output / 'resumption_implementation' / 'adt' / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(raw)
            _json(output / 'resumption.json', {**resumption,
                'reason': 'replace_progress_checkpoint_instead_of_create_only',
                'completed_subjects_reused': list(reports),
                'integrations_reused': len(reports) * len(plan.arms) * len(VARIANTS),
                'resumed_utc': datetime.now(timezone.utc).isoformat()})
    group, _ = plan.select(batch)
    if resume_dir is None:
        _json(output / 'inputs.json', {
        'profile': plan.profile, 'model_version': '0.3', 'batch': batch,
        'batch_count': plan.batches, 'phenotype': group, 'case_ids': ids,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': plan.config_hash, 'interventions_sha256': plan.catalog_hash,
        'pilot_dir': str(plan.pilot_dir), 'pilot_summary_sha256': plan.pilot_summary_hash,
        'pilot_inputs_sha256': plan.pilot_inputs_hash,
        'source_sha256': source_hashes,
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy', 'scipy', 'PyYAML')},
        'state_order': STATE_NAMES, 'starting_states': starting,
        'arms': {name: asdict(arm) for name, arm in plan.arms.items()},
        'checkpoints': plan.checkpoints, 'agreement': asdict(plan.agreement),
        'primary_metric': 'AUC_A42', 'primary_checkpoint': plan.primary_checkpoint,
        'responder_classification': 'continuous_only',
        })
    total = len(ids) * len(plan.arms) * len(VARIANTS)
    reused_integrations = len(reports) * len(plan.arms) * len(VARIANTS)
    progress = None if not show_progress else ProgressBar(total, label=f'P12 lote {batch}/{plan.batches}',
        unit='integraciones', interval=interval)
    if progress is not None:
        progress.done = reused_integrations
        progress.start()
    try:
        for case_id in ids:
            if case_id in reports:
                continue
            case = plan.cases[case_id]
            parameters = ModelParameters(**case['parameters'])
            subject = SubjectParameters(**case['subject'])
            external = ConstantInput(**case['external_input'])
            initial = state_vector([starting[case_id]['state'][name] for name in STATE_NAMES])
            arms = {}
            comparisons = {}
            issues = []
            for arm_name, intervention in plan.arms.items():
                runs = {}
                for variant in VARIANTS:
                    phase = f'{case_id}/{arm_name}/{variant}'
                    if progress is not None:
                        progress.stage(f'{phase}: integrando')
                    run = _run_arm(parameters, subject, external, intervention,
                                   initial, plan.settings[variant], plan.checkpoints)
                    runs[variant] = run
                    _save_run(output, f'{case_id}/{arm_name}', variant, run, echo=False)
                    if progress is not None:
                        progress.advance(f'{phase}: {run.status}; archivo guardado')
                    if run.status != 'completed':
                        issues.append(f'{arm_name}/{variant}: {run.message}')
                arms[arm_name] = runs
                comparisons[arm_name] = {label: _compare(runs[left], runs[right], plan.agreement)
                                         for label, left, right in COMPARISONS}
                issues.extend(f'{arm_name}/{label}: numerical_agreement_failed'
                              for label, result in comparisons[arm_name].items()
                              if result['passes'] is not True)
            control = arms['control']['main'].outcomes if arms['control']['main'].status == 'completed' else None
            case_rows = []
            for arm_name, intervention in plan.arms.items():
                if arm_name == 'control':
                    continue
                treatment = arms[arm_name]['main']
                if treatment.status != 'completed':
                    continue
                for row in contrast_outcomes(treatment.outcomes, control):
                    if row['metric'] not in ('AUC_A42', *SUPPLEMENTARY):
                        continue
                    case_rows.append({'case_id': case_id, 'phenotype': group,
                        'arm': arm_name, 'kind': intervention.kind,
                        'strength': intervention.strength, 'checkpoint': row['checkpoint'],
                        'metric': row['metric'],
                        'is_primary': row['metric'] == 'AUC_A42'
                                      and row['checkpoint'] == plan.primary_checkpoint,
                        'value': row['value'], 'control_value': row['control_value'],
                        'difference_from_control': row['difference_from_control'],
                        'percent_change_from_control': row['percent_change_from_control'],
                        'contrast_reason': row['contrast_reason']})
            _write_contrasts(output / 'cases' / case_id / 'paired_contrasts.csv', case_rows)
            rows.extend(case_rows)
            issues.extend(f'{row["arm"]}/{row["metric"]}@{row["checkpoint"]}: {row["contrast_reason"]}'
                          for row in case_rows if row['contrast_reason'] is not None
                          and row['metric'] == 'AUC_A42'
                          and row['checkpoint'] == plan.primary_checkpoint)
            report = {'case_id': case_id, 'phenotype': group,
                      'c1_main_run_sha256': starting[case_id]['run_sha256'],
                      'initial_state': starting[case_id]['state'],
                      'arms': {name: {variant: {'status': run.status, 'message': run.message}
                                      for variant, run in runs.items()}
                               for name, runs in arms.items()},
                      'comparisons': comparisons,
                      'numerically_confirmed': not issues, 'issues': issues,
                      'responder_classification': 'not_assessed'}
            reports[case_id] = report
            _json(output / 'cases' / case_id / 'decision.json', report)
            _save_progress(output / 'progress.json', {
                'last_case': case_id, 'subjects_attempted': len(reports),
                'subjects_numerically_confirmed': sum(x['numerically_confirmed'] for x in reports.values()),
                'integrations_attempted': len(reports) * len(plan.arms) * len(VARIANTS),
            })
    finally:
        if progress is not None:
            progress.close()
    _write_contrasts(output / 'paired_contrasts.csv', rows)
    confirmed = len(reports) == len(ids) and all(x['numerically_confirmed'] for x in reports.values())
    summary = {'profile': plan.profile, 'status': 'confirmed' if confirmed else 'not_confirmed',
               'numerically_confirmed': confirmed, 'continuous_analysis_only': True,
               'biological_status': 'unclassified', 'responder_classification': 'not_assessed',
               'batch': batch, 'batch_count': plan.batches, 'phenotype': group,
               'subject_ids': ids, 'subjects_planned': len(ids), 'subjects_attempted': len(reports),
               'subjects_numerically_confirmed': sum(x['numerically_confirmed'] for x in reports.values()),
               'integrations_planned': total,
               'integrations_attempted': len(reports) * len(plan.arms) * len(VARIANTS),
               'integrations_reused': reused_integrations,
               'resumption_record': 'resumption.json' if resume_dir is not None else None,
               'arms_per_subject': len(plan.arms), 'variants_per_arm': len(VARIANTS),
               'pilot_summary_sha256': plan.pilot_summary_hash,
               'pilot_inputs_sha256': plan.pilot_inputs_hash,
               'config_sha256': plan.config_hash, 'interventions_sha256': plan.catalog_hash,
               'issues': [f'{case_id}: {issue}' for case_id, report in reports.items()
                          for issue in report['issues']],
               'contrasts_file': 'paired_contrasts.csv',
               'finished_utc': datetime.now(timezone.utc).isoformat()}
    _json(output / 'summary.json', summary)
    print(f'\n{summary["status"]}: {summary["subjects_numerically_confirmed"]}/{len(ids)} '
          f'sujetos; {summary["integrations_attempted"]}/{total} integraciones intentadas')
    print(f'Artifacts: {output}')
    return 0 if confirmed else 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one selected 20-subject GSI/GSM paired batch.')
    parser.add_argument('--config', type=Path, default=Path('config/interventions_paired.yaml'))
    parser.add_argument('--batch', type=int, required=True, help='Batch number, 1 through 6.')
    parser.add_argument('--check-only', action='store_true', help='Read-only preflight; no RHS or files.')
    parser.add_argument('--resume-dir', type=Path,
                        help='Continue an interrupted batch; reuse its completed subjects.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan, ids, config_raw, catalog_raw, starting = _preflight(args.config, args.batch)
        if not 0 < args.progress_interval < float('inf'):
            raise ValueError('--progress-interval must be positive and finite.')
        reused = 0
        if args.resume_dir is not None:
            hashes = {name: sha256(raw) for name, raw in _source_files().items()}
            reused = len(_resume_completed(plan, ids, args.batch,
                                           args.resume_dir.resolve(), starting, hashes)[0])
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    group, _ = plan.select(args.batch)
    total = len(ids) * len(plan.arms) * len(VARIANTS)
    if args.check_only:
        print(f'P12 preflight OK: lote {args.batch}/{plan.batches}, {group}, '
              f'{ids[0]}–{ids[-1]}, {len(ids)} sujetos, {total} integraciones previstas; '
              f'{reused * len(plan.arms) * len(VARIANTS)} integraciones reutilizables.')
        print('Sin RHS, integración ni archivos nuevos.')
        return 0
    return _execute(plan, ids, args.batch, config_raw, catalog_raw,
                    starting, args.progress_interval, not args.no_progress, args.resume_dir)
