"""Render the frozen P15 G8 publication package without new simulations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

from .basal_cli import _json
from .parameters import load_config, mapping
from .progress import ProgressBar
from .reference_artifacts import sha256
from .therapeutic_surrogate import PRIMARY_TARGET, TREATED_ARMS


FIGURE_IDS = (
    'figure_1_architecture',
    'figure_2_joint_design',
    'figure_3_response_landscape',
    'figure_4_heterogeneity',
    'figure_5_surrogate_fidelity',
    'figure_6_domain_speed',
)
FORMATS = ('svg', 'pdf', 'png')


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


def _csv_bytes(fields: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob('*.py'))}


@dataclass(frozen=True)
class PublicationPlan:
    raw: bytes
    config_hash: str
    output_root: Path
    formats: tuple[str, ...]
    dpi: int
    font_family: str
    figure_ids: tuple[str, ...]
    jt1_dir: Path
    dataset_dir: Path
    selection_dir: Path
    evaluation_dir: Path
    source_raw: Mapping[str, bytes]
    jt1_summary: dict
    dataset_summary: dict
    selection_summary: dict
    evaluation_summary: dict
    opening: dict
    acceptance: dict
    domain: dict
    rejections: dict
    performance: dict


def _load_plan(path: Path) -> PublicationPlan:
    raw = path.read_bytes()
    data = load_config(path, allow_incomplete=False).data
    _require(set(data) == {'profile', 'status', 'source', 'render', 'figures'},
             'Unexpected P15 G8 configuration fields.')
    _require(data['profile'] == 'P15_G8' and data['status'] == 'selected',
             'Publication configuration must be the selected P15_G8 profile.')
    source = mapping(data['source'], 'source')
    expected_source = {
        'jt1_collection', 'jt1_summary_sha256', 'jt1_subjects_sha256',
        'jt1_contrasts_sha256', 'jt1_descriptive_sha256',
        'jt1_agreement_sha256', 'dataset_dir',
        'dataset_summary_sha256', 'domain_sha256', 'selection_dir',
        'selection_summary_sha256', 'selections_sha256', 'evaluation_dir',
        'evaluation_summary_sha256', 'opening_sha256',
        'internal_predictions_sha256', 'internal_metrics_sha256',
        'internal_metrics_json_sha256', 'acceptance_sha256',
        'external_rejections_sha256', 'performance_sha256',
    }
    _require(set(source) == expected_source,
             'P15 G8 source lineage is incomplete or contains unknown fields.')

    jt1_dir = Path(source['jt1_collection']).resolve()
    dataset_dir = Path(source['dataset_dir']).resolve()
    selection_dir = Path(source['selection_dir']).resolve()
    evaluation_dir = Path(source['evaluation_dir']).resolve()
    roots = (
        (jt1_dir, Path('results/interventions/joint_final/collection').resolve()),
        (dataset_dir, Path('results/surrogate/therapeutic/dataset').resolve()),
        (selection_dir, Path('results/surrogate/therapeutic/selection').resolve()),
        (evaluation_dir, Path('results/surrogate/therapeutic/evaluation').resolve()),
    )
    _require(all(directory.is_relative_to(root) for directory, root in roots),
             'A publication source is outside its approved results root.')

    files = {
        'jt1_summary': (jt1_dir / 'summary.json', source['jt1_summary_sha256']),
        'jt1_subjects': (jt1_dir / 'subjects.csv', source['jt1_subjects_sha256']),
        'jt1_contrasts': (
            jt1_dir / 'paired_contrasts.csv', source['jt1_contrasts_sha256']),
        'jt1_descriptive': (
            jt1_dir / 'descriptive_summary.csv', source['jt1_descriptive_sha256']),
        'jt1_agreement': (
            jt1_dir / 'numerical_agreement.csv', source['jt1_agreement_sha256']),
        'dataset_summary': (
            dataset_dir / 'summary.json', source['dataset_summary_sha256']),
        'domain': (dataset_dir / 'domain.json', source['domain_sha256']),
        'selection_summary': (
            selection_dir / 'summary.json', source['selection_summary_sha256']),
        'selections': (selection_dir / 'selections.csv', source['selections_sha256']),
        'evaluation_summary': (
            evaluation_dir / 'summary.json', source['evaluation_summary_sha256']),
        'opening': (evaluation_dir / 'opening.json', source['opening_sha256']),
        'internal_predictions': (
            evaluation_dir / 'internal_predictions.csv',
            source['internal_predictions_sha256']),
        'internal_metrics': (
            evaluation_dir / 'internal_metrics.csv', source['internal_metrics_sha256']),
        'internal_metrics_json': (
            evaluation_dir / 'internal_metrics.json',
            source['internal_metrics_json_sha256']),
        'acceptance': (
            evaluation_dir / 'acceptance.json', source['acceptance_sha256']),
        'external_rejections': (
            evaluation_dir / 'external_domain_rejections.json',
            source['external_rejections_sha256']),
        'performance': (
            evaluation_dir / 'performance.json', source['performance_sha256']),
    }
    source_raw = {name: _pinned(file_path, digest, name)
                  for name, (file_path, digest) in files.items()}
    jt1 = _json_object(source_raw['jt1_summary'], 'JT1 summary')
    dataset = _json_object(source_raw['dataset_summary'], 'E1 dataset summary')
    selection = _json_object(source_raw['selection_summary'], 'E1 selection summary')
    evaluation = _json_object(source_raw['evaluation_summary'], 'E1 evaluation summary')
    opening = _json_object(source_raw['opening'], 'E1 opening')
    acceptance = _json_object(source_raw['acceptance'], 'E1 acceptance')
    domain = _json_object(source_raw['domain'], 'E1 domain')
    rejections = _json_object(source_raw['external_rejections'], 'P12 rejections')
    performance = _json_object(source_raw['performance'], 'E1 performance')

    _require(jt1.get('profile') == 'P15_JT1_A2FULL'
             and jt1.get('status') == 'confirmed'
             and jt1.get('numerically_confirmed') is True
             and jt1.get('subjects_selected') == 1024
             and jt1.get('contrast_rows') == 229376
             and jt1.get('agreement_rows') == jt1.get('agreement_rows_passing') == 16528,
             'G8 requires the fully confirmed JT1 collection.')
    _require(dataset.get('stage') == 'dataset' and dataset.get('status') == 'prepared'
             and dataset.get('subjects') == {'train': 640, 'validation': 192,
                                               'test': 192}
             and dataset.get('rows') == {'train': 143360, 'validation': 43008,
                                          'test': 43008}
             and dataset.get('targets') == 32,
             'G8 requires the frozen E1 dataset and subject-level split.')
    _require(selection.get('stage') == 'selection'
             and selection.get('status') == 'selected'
             and selection.get('models_frozen') is True
             and selection.get('test_open_allowed') is True
             and selection.get('targets_selected') == 32
             and selection.get('fits_attempted') == 736,
             'G8 requires the frozen 32-target E1 selection.')
    _require(evaluation.get('stage') == 'evaluation'
             and evaluation.get('status') == 'accepted'
             and evaluation.get('test_opened_once') is True
             and evaluation.get('retuning_after_test') is False
             and evaluation.get('primary_passed') is True
             and evaluation.get('secondary_passed') is True
             and evaluation.get('internal_rows') == 43008,
             'G8 requires the accepted single-opening E1 evaluation.')
    _require(opening.get('policy') == 'explicit_once_after_frozen_selection'
             and opening.get('selection_summary_sha256')
                 == source['selection_summary_sha256'],
             'The archived E1 test-opening record changed.')
    _require(acceptance.get('accepted_for_within_domain_screening') is True,
             'E1 was not accepted for within-domain screening.')
    _require(rejections.get('rows') == 26880
             and rejections.get('in_domain_rows') == 0
             and rejections.get('hard_acceptance_gate') is False,
             'P12 must remain a descriptive out-of-domain applicability check.')

    render = mapping(data['render'], 'render')
    _require(set(render) == {'formats', 'dpi', 'font_family', 'style',
                             'language', 'output_root'}
             and tuple(render['formats']) == FORMATS
             and render['style'] == 'colorblind_mechanism_separated'
             and render['language'] == 'en'
             and isinstance(render['dpi'], int) and render['dpi'] >= 300,
             'Unexpected P15 G8 render specification.')
    output_root = Path(render['output_root']).resolve()
    _require(output_root.is_relative_to(Path('results/publication/stage15').resolve()),
             'Publication outputs must remain under results/publication/stage15.')
    figure_ids = tuple(mapping(item, 'figure')['id'] for item in data['figures'])
    _require(figure_ids == FIGURE_IDS, 'The six prespecified G8 figures changed.')
    return PublicationPlan(
        raw=raw, config_hash=sha256(raw), output_root=output_root,
        formats=FORMATS, dpi=int(render['dpi']),
        font_family=str(render['font_family']), figure_ids=figure_ids,
        jt1_dir=jt1_dir, dataset_dir=dataset_dir,
        selection_dir=selection_dir, evaluation_dir=evaluation_dir,
        source_raw=source_raw, jt1_summary=jt1, dataset_summary=dataset,
        selection_summary=selection, evaluation_summary=evaluation,
        opening=opening, acceptance=acceptance, domain=domain,
        rejections=rejections, performance=performance)


def _load_rows(plan: PublicationPlan) -> tuple[
        list[dict[str, str]], list[dict[str, str]], list[dict[str, str]],
        list[dict[str, str]], list[dict[str, str]], list[dict]]:
    subjects = _read_csv(plan.jt1_dir / 'subjects.csv')
    contrasts = _read_csv(plan.jt1_dir / 'paired_contrasts.csv')
    selections = _read_csv(plan.selection_dir / 'selections.csv')
    predictions = _read_csv(plan.evaluation_dir / 'internal_predictions.csv')
    metrics = _read_csv(plan.evaluation_dir / 'internal_metrics.csv')
    details = json.loads(plan.source_raw['internal_metrics_json'])
    _require(len(subjects) == 1024 and len(contrasts) == 229376,
             'JT1 figure row counts changed.')
    _require(len(selections) == 32 and len(metrics) == 32
             and isinstance(details, list) and len(details) == 32,
             'E1 target artifacts must contain 32 complete targets.')
    _require(len(predictions) == 43008
             and sum(row['target_id'] == PRIMARY_TARGET for row in predictions) == 1344,
             'E1 prediction row counts changed.')
    _require(all(row['status'] == 'selected' for row in selections)
             and all(row['status'] == 'evaluated' for row in metrics),
             'G8 cannot render incomplete selection or metric rows.')
    return subjects, contrasts, selections, predictions, metrics, details


def _table_rows(plan: PublicationPlan, selections: list[dict[str, str]],
                metrics: list[dict[str, str]], details: list[dict]) -> dict[str, bytes]:
    from .stage15_publication import ARM_LABELS

    primary = next(row for row in metrics if row['target_id'] == PRIMARY_TARGET)
    primary_detail = next(row for row in details if row['target_id'] == PRIMARY_TARGET)
    thresholds = mapping(plan.acceptance['primary_thresholds'], 'primary thresholds')
    checks = mapping(plan.acceptance['primary_checks'], 'primary checks')
    primary_specs = (
        ('all_test_targets_defined', 'n_defined = n_total', primary['n_defined'],
         checks['all_test_targets_defined']),
        ('global_nrmse', f"<= {thresholds['global_nrmse_max']}",
         primary['global_nrmse'], checks['global_nrmse']),
        ('macro_arm_nrmse', f"<= {thresholds['macro_arm_nrmse_max']}",
         primary['macro_arm_nrmse'], checks['macro_arm_nrmse']),
        ('worst_arm_nrmse', f"<= {thresholds['each_arm_nrmse_max']}",
         primary['worst_arm_nrmse'], checks['each_arm_nrmse']),
        ('p95_over_global_scale',
         f"<= {thresholds['p95_absolute_error_over_global_scale_max']}",
         primary['p95_over_global_scale'], checks['p95_absolute_error']),
        ('sign_concordance', f">= {thresholds['sign_concordance_min']}",
         primary['sign_concordance'], checks['sign_concordance']),
        ('boundary_nrmse', f"<= {thresholds['boundary_nrmse_max']}",
         primary['boundary_nrmse'], checks['boundary_nrmse']),
        ('domain_coverage', f">= {thresholds['internal_domain_coverage_min']}",
         primary['domain_coverage'], checks['domain_coverage']),
        ('finite_predictions', 'all finite', primary['finite_predictions'],
         checks['finite_predictions']),
    )
    protocol_rows = [
        {'section': 'JT1', 'item': 'subjects',
         'value': plan.jt1_summary['subjects_selected']},
        {'section': 'JT1', 'item': 'joint_cells',
         'value': plan.jt1_summary['cells']},
        {'section': 'JT1', 'item': 'arms_per_subject',
         'value': plan.jt1_summary['arms_per_subject']},
        {'section': 'JT1', 'item': 'numerical_variants',
         'value': plan.jt1_summary['variants_per_arm']},
        {'section': 'JT1', 'item': 'archived_integrations',
         'value': plan.jt1_summary['integrations_archived']},
        {'section': 'E1', 'item': 'train_subjects',
         'value': plan.dataset_summary['subjects']['train']},
        {'section': 'E1', 'item': 'validation_subjects',
         'value': plan.dataset_summary['subjects']['validation']},
        {'section': 'E1', 'item': 'test_subjects',
         'value': plan.dataset_summary['subjects']['test']},
        {'section': 'E1', 'item': 'independent_targets',
         'value': plan.dataset_summary['targets']},
        {'section': 'E1', 'item': 'candidate_fits',
         'value': plan.selection_summary['fits_attempted']},
        {'section': 'E1', 'item': 'internal_test_rows',
         'value': plan.evaluation_summary['internal_rows']},
    ]
    acceptance_rows = [
        {'criterion': name, 'prespecified_threshold': threshold,
         'observed': observed, 'passed': passed}
        for name, threshold, observed, passed in primary_specs]
    secondary = mapping(plan.acceptance['secondary_summary'], 'secondary summary')
    benchmark = mapping(plan.performance['surrogate_benchmark'], 'benchmark')
    summary_rows = [
        {'category': 'secondary', 'item': key, 'value': value}
        for key, value in secondary.items()
    ] + [
        {'category': 'performance', 'item': 'batch_seconds_per_32_target_request',
         'value': benchmark['batch_seconds_per_32_target_request']},
        {'category': 'performance', 'item': 'operational_speedup',
         'value': benchmark['operational_speedup_vs_archived_integration_batch']},
        {'category': 'applicability', 'item': 'P12_rows',
         'value': plan.rejections['rows']},
        {'category': 'applicability', 'item': 'P12_in_domain_rows',
         'value': plan.rejections['in_domain_rows']},
    ]
    by_arm = mapping(primary_detail['metrics'], 'primary metrics')['by_arm']
    arm_rows = [
        {'arm': arm, 'label': ARM_LABELS[arm], **mapping(by_arm[arm], arm)}
        for arm in TREATED_ARMS]
    provenance_rows = []
    source_paths = {
        'jt1_summary': plan.jt1_dir / 'summary.json',
        'jt1_subjects': plan.jt1_dir / 'subjects.csv',
        'jt1_contrasts': plan.jt1_dir / 'paired_contrasts.csv',
        'jt1_descriptive': plan.jt1_dir / 'descriptive_summary.csv',
        'jt1_agreement': plan.jt1_dir / 'numerical_agreement.csv',
        'dataset_summary': plan.dataset_dir / 'summary.json',
        'domain': plan.dataset_dir / 'domain.json',
        'selection_summary': plan.selection_dir / 'summary.json',
        'selections': plan.selection_dir / 'selections.csv',
        'evaluation_summary': plan.evaluation_dir / 'summary.json',
        'opening': plan.evaluation_dir / 'opening.json',
        'internal_predictions': plan.evaluation_dir / 'internal_predictions.csv',
        'internal_metrics': plan.evaluation_dir / 'internal_metrics.csv',
        'internal_metrics_json': plan.evaluation_dir / 'internal_metrics.json',
        'acceptance': plan.evaluation_dir / 'acceptance.json',
        'external_rejections': plan.evaluation_dir / 'external_domain_rejections.json',
        'performance': plan.evaluation_dir / 'performance.json',
    }
    for name, raw in plan.source_raw.items():
        provenance_rows.append({'source': name, 'path': _relative(source_paths[name]),
                                'sha256': sha256(raw)})
    p12_rows = [
        {'record_type': 'summary', 'name': 'total_rows',
         'lower': '', 'upper': '', 'count': plan.rejections['rows']},
        {'record_type': 'summary', 'name': 'in_domain_rows',
         'lower': '', 'upper': '', 'count': plan.rejections['in_domain_rows']},
    ]
    for name, bounds in mapping(plan.domain['factor_ranges'], 'factor ranges').items():
        p12_rows.append({'record_type': 'factor_domain', 'name': name,
                         'lower': bounds[0], 'upper': bounds[1], 'count': ''})
    for reason, count in mapping(plan.rejections['rejections'], 'rejections').items():
        p12_rows.append({'record_type': 'first_rejection', 'name': reason,
                         'lower': '', 'upper': '', 'count': count})
    return {
        'table_1_protocol.csv': _csv_bytes(
            ('section', 'item', 'value'), protocol_rows),
        'table_2_primary_acceptance.csv': _csv_bytes(
            ('criterion', 'prespecified_threshold', 'observed', 'passed'),
            acceptance_rows),
        'table_3_secondary_performance.csv': _csv_bytes(
            ('category', 'item', 'value'), summary_rows),
        'table_s1_internal_metrics.csv': plan.source_raw['internal_metrics'],
        'table_s2_selections.csv': plan.source_raw['selections'],
        'table_s3_primary_by_arm.csv': _csv_bytes(
            ('arm', 'label', 'n', 'mae', 'rmse', 'nrmse'), arm_rows),
        'table_s4_provenance.csv': _csv_bytes(
            ('source', 'path', 'sha256'), provenance_rows),
        'table_s5_p12_domain.csv': _csv_bytes(
            ('record_type', 'name', 'lower', 'upper', 'count'), p12_rows),
        'table_s6_numerical_audit.csv': plan.source_raw['jt1_agreement'],
    }


def _render(plan: PublicationPlan, progress_interval: float) -> Path:
    from .stage15_publication import (
        figure_1_architecture,
        figure_2_joint_design,
        figure_3_response_landscape,
        figure_4_heterogeneity,
        figure_5_surrogate_fidelity,
        figure_6_domain_speed,
        save_figure,
    )

    subjects, contrasts, selections, predictions, metrics, details = _load_rows(plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = plan.output_root / stamp
    figures_dir = output / 'figures'
    tables_dir = output / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=False)
    tables_dir.mkdir()
    (output / 'stage15_publication.yaml').write_bytes(plan.raw)
    _json(output / 'run.json', {
        'profile': 'P15_G8', 'stage': 'publication', 'status': 'rendering',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': plan.config_hash,
        'source_sha256': {name: sha256(raw) for name, raw in plan.source_raw.items()},
    })

    factories = (
        lambda: figure_1_architecture(plan.font_family),
        lambda: figure_2_joint_design(subjects, plan.font_family),
        lambda: figure_3_response_landscape(contrasts, plan.font_family),
        lambda: figure_4_heterogeneity(contrasts, plan.font_family),
        lambda: figure_5_surrogate_fidelity(
            predictions, metrics, selections, plan.font_family),
        lambda: figure_6_domain_speed(
            plan.domain, plan.rejections, plan.performance, plan.font_family),
    )
    progress = ProgressBar(6, label='P15 G8', unit='figuras',
                           interval=progress_interval)
    generated: list[Path] = []
    progress.start()
    try:
        for index, (identifier, factory) in enumerate(
                zip(plan.figure_ids, factories, strict=True), start=1):
            progress.stage(f'figura {index}/6: {identifier}')
            generated.extend(save_figure(
                factory(), figures_dir / identifier, plan.formats, plan.dpi))
            progress.advance(f'figura {index}/6 guardada')
    finally:
        progress.close()

    tables = _table_rows(plan, selections, metrics, details)
    for name, raw in tables.items():
        (tables_dir / name).write_bytes(raw)
    sources = _source_files()
    for name, raw in sources.items():
        destination = output / 'implementation' / 'adt' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    script = Path('scripts/render_stage15_figures.py')
    destination = output / 'implementation' / 'scripts' / script.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(script.read_bytes())

    archived = [path for path in output.rglob('*') if path.is_file()
                and path.name not in {'run.json', 'summary.json'}]
    summary = {
        'profile': 'P15_G8', 'stage': 'publication', 'status': 'rendered',
        'scientific_results_recomputed': False,
        'figures': 6, 'formats_per_figure': len(plan.formats),
        'figure_files': len(generated), 'tables': len(tables),
        'config_sha256': plan.config_hash,
        'source_sha256': {name: sha256(raw) for name, raw in plan.source_raw.items()},
        'artifact_sha256': {
            file_path.relative_to(output).as_posix(): sha256(file_path.read_bytes())
            for file_path in archived},
        'packages': {'python': platform.python_version(),
                     'numpy': version('numpy'),
                     'PyYAML': version('PyYAML'),
                     'matplotlib': version('matplotlib')},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _json(output / 'summary.json', summary)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Render the frozen P15 G8 publication figures and tables.')
    parser.add_argument('--config', type=Path,
                        default=Path('config/stage15_publication.yaml'))
    parser.add_argument('--check-only', action='store_true',
                        help='Verify all pinned sources without writing figures.')
    parser.add_argument('--progress-interval', type=float, default=5.)
    args = parser.parse_args(argv)
    _require(args.progress_interval > 0., 'Progress interval must be positive.')
    try:
        version('matplotlib')
    except PackageNotFoundError as error:
        raise ValueError(
            'Matplotlib is required; install the publication extra with '
            'python -m pip install -e ".[publication]".') from error
    plan = _load_plan(args.config)
    if args.check_only:
        _load_rows(plan)
        print('P15 G8 preflight OK: fuentes JT1/E1 fijadas y 6 figuras verificadas.')
        print('Plan: 6 figuras × 3 formatos y 9 tablas; sin simulación ni reajuste.')
        print('Sin escritura de archivos nuevos.')
        return 0
    output = _render(plan, args.progress_interval)
    print('rendered: 6 figuras, 18 archivos gráficos y 9 tablas; resultados no recalculados')
    print(f'Artifacts: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
