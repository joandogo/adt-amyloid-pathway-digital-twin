"""Static scientific figures for the frozen P15 JT1+E1 publication package."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np

from .therapeutic_surrogate import CHECKPOINTS, METRICS, PRIMARY_TARGET, TREATED_ARMS


ARM_LABELS = {
    'GSI_01': 'GSI 0.10', 'GSI_02': 'GSI 0.25',
    'GSI_03': 'GSI 0.50', 'GSI_04': 'GSI 0.75',
    'GSM_low': 'GSM 0.10', 'GSM_medium': 'GSM 0.25',
    'GSM_high': 'GSM 0.50',
}
ARM_COLORS = {
    'GSI_01': '#b9d6ea', 'GSI_02': '#74a9cf',
    'GSI_03': '#2b8cbe', 'GSI_04': '#045a8d',
    'GSM_low': '#fdd49e', 'GSM_medium': '#fdbb84',
    'GSM_high': '#e34a33',
}
SPLIT_COLORS = {'train': '#4c78a8', 'validation': '#f58518', 'test': '#7a5195'}
METRIC_LABELS = {
    'AUC_A42': 'AUC Aβ42', 'A42_final': 'Final Aβ42',
    'Ratio_42_40': 'Aβ42/Aβ40', 'AUC_O': 'AUC oligomers',
    'P_final': 'Final plaques', 'AUC_I': 'AUC inflammation',
    'C99_max': 'Maximum C99', 'AUC_C99': 'AUC C99',
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(document, dict), f'{path}: expected a JSON object.')
    return document


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-.10, 1.08, label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='right')


def style(font_family: str) -> dict:
    return {
        'font.family': font_family,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.bbox': 'tight',
    }


def save_figure(figure: plt.Figure, base: Path, formats: Sequence[str],
                dpi: int) -> list[Path]:
    paths = []
    for extension in formats:
        path = base.with_suffix(f'.{extension}')
        figure.savefig(path, dpi=dpi if extension == 'png' else None,
                       facecolor='white')
        paths.append(path)
    plt.close(figure)
    return paths


def _box(ax: plt.Axes, x: float, y: float, width: float, height: float,
         text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle='round,pad=0.012,rounding_size=0.015',
        linewidth=1.2, edgecolor=color, facecolor='white')
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center')


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           color: str = '#555555') -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                linewidth=1.1, color=color))


def figure_1_architecture(font_family: str) -> plt.Figure:
    with plt.rc_context(style(font_family)):
        figure, ax = plt.subplots(figsize=(12, 6.6), constrained_layout=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        _box(ax, .03, .73, .10, .10, 'APP', '#4c78a8')
        _box(ax, .19, .73, .10, .10, 'C99', '#4c78a8')
        _box(ax, .35, .73, .15, .10, 'Aβ40 / Aβ42', '#4c78a8')
        _box(ax, .58, .84, .13, .10, 'Oligomers', '#4c78a8')
        _box(ax, .58, .62, .13, .10, 'Plaques', '#4c78a8')
        _box(ax, .83, .73, .14, .10, 'Inflammation', '#4c78a8')
        _arrow(ax, (.13, .78), (.19, .78))
        _arrow(ax, (.29, .78), (.35, .78))
        _arrow(ax, (.50, .79), (.58, .88))
        _arrow(ax, (.50, .77), (.58, .68))
        _arrow(ax, (.645, .84), (.645, .72))
        _arrow(ax, (.71, .89), (.83, .80))
        _arrow(ax, (.71, .67), (.83, .76))
        ax.add_patch(FancyArrowPatch(
            (.88, .84), (.15, .83), arrowstyle='-|>', mutation_scale=12,
            linewidth=1.1, color='#7a5195',
            connectionstyle='arc3,rad=.18'))
        ax.text(.60, .975, 'O/P-driven inflammation enhances BACE processing',
                color='#7a5195', ha='center', fontweight='bold')
        ax.text(.27, .90, 'GSI: reduced γ-secretase activity', color='#045a8d',
                ha='center', fontweight='bold')
        ax.annotate('', xy=(.35, .84), xytext=(.27, .89),
                    arrowprops={'arrowstyle': '-|>', 'color': '#045a8d'})
        ax.text(.39, .62, 'GSM: shifted product fractions; γ activity unchanged',
                color='#e34a33', ha='center', fontweight='bold')
        ax.annotate('', xy=(.42, .72), xytext=(.39, .64),
                    arrowprops={'arrowstyle': '-|>', 'color': '#e34a33'})

        flow = [
            ('P14 joint cohort\n10,000 controls', .03, '#777777'),
            ('JT1 paired trial\n1,024 subjects × 8 arms', .23, '#4c78a8'),
            ('Subject-level split\n640 / 192 / 192', .47, '#7a5195'),
            ('32 frozen surrogates\n736 candidate fits', .68, '#f58518'),
            ('Single test opening\nwithin-domain screening', .835, '#54a24b'),
        ]
        widths_flow = [.15, .19, .17, .14, .145]
        for (text, x, color), width in zip(flow, widths_flow, strict=True):
            _box(ax, x, .25, width, .14, text, color)
        for index in range(len(flow) - 1):
            _arrow(ax, (flow[index][1] + widths_flow[index], .32),
                   (flow[index + 1][1], .32))
        ax.text(.50, .08,
                'Numerical audit → paired mechanistic responses → frozen domain guard',
                ha='center', fontsize=11, fontweight='bold')
        panel(ax, 'A')
        figure.suptitle('Mechanistic-to-surrogate pathway-level digital twin',
                        fontsize=14, fontweight='bold')
        return figure


def figure_2_joint_design(subjects: list[dict[str, str]],
                          font_family: str) -> plt.Figure:
    _require(len(subjects) == 1024, 'Figure 2 requires 1,024 JT1 subjects.')
    with plt.rc_context(style(font_family)):
        figure = plt.figure(figsize=(13, 8), constrained_layout=True)
        grid = figure.add_gridspec(2, 4)
        scatter = figure.add_subplot(grid[0, :2])
        heat_axes = [figure.add_subplot(grid[row, column])
                     for row in range(2) for column in (2, 3)]
        split_ax = figure.add_subplot(grid[1, :2])

        markers = {'train': 'o', 'validation': 's', 'test': '^'}
        for split in ('train', 'validation', 'test'):
            rows = [row for row in subjects if row['split'] == split]
            points = scatter.scatter(
                [float(row['k_BACE_multiplier']) for row in rows],
                [float(row['C_i']) for row in rows],
                c=[float(row['I_ext']) for row in rows], cmap='viridis',
                vmin=.25, vmax=1., marker=markers[split], s=19,
                alpha=.72, linewidths=0, label=split.title())
        scatter.set(xlabel=r'$k_{BACE}$ multiplier', ylabel=r'$C_i$',
                    title='Joint factor coverage and subject-level split')
        scatter.legend(frameon=False, ncol=3)
        figure.colorbar(points, ax=scatter, label=r'$I_{ext}$')
        panel(scatter, 'A')

        for i_ext_bin, ax in enumerate(heat_axes, start=1):
            matrix = np.zeros((4, 4), dtype=int)
            for row in subjects:
                if int(row['bin_I_ext']) == i_ext_bin:
                    x = int(row['bin_k_BACE_multiplier']) - 1
                    y = int(row['bin_C_i']) - 1
                    matrix[y, x] += 1
            image = ax.imshow(matrix, origin='lower', vmin=0, vmax=16,
                              cmap='Blues')
            for y in range(4):
                for x in range(4):
                    ax.text(x, y, str(matrix[y, x]), ha='center', va='center',
                            color='white' if matrix[y, x] > 9 else '#222222')
            ax.set(title=rf'$I_{{ext}}$ bin {i_ext_bin}',
                   xlabel=r'$k_{BACE}$ bin', ylabel=r'$C_i$ bin',
                   xticks=range(4), yticks=range(4))
            ax.set_xticklabels(range(1, 5))
            ax.set_yticklabels(range(1, 5))
        figure.colorbar(image, ax=heat_axes, label='Subjects per cell', shrink=.75)
        panel(heat_axes[0], 'B')

        counts = Counter(row['split'] for row in subjects)
        names = ['train', 'validation', 'test']
        bars = split_ax.bar(names, [counts[name] for name in names],
                            color=[SPLIT_COLORS[name] for name in names])
        split_ax.bar_label(bars)
        split_ax.set(ylabel='Subjects', title='Frozen partition')
        split_ax.text(.98, .88,
                      '8 arms × 3 numerical variants\n4 checkpoints × 8 metrics',
                      transform=split_ax.transAxes, ha='right', va='top',
                      bbox={'boxstyle': 'round', 'facecolor': 'white',
                            'edgecolor': '#777777'})
        panel(split_ax, 'C')
        figure.suptitle('Joint JT1 design and balanced coverage',
                        fontsize=14, fontweight='bold')
        return figure


def _defined_contrasts(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows
            if row['contrast_reason'] == '' and row['percent_change_from_control'] != '']


def _colored_boxplot(ax: plt.Axes, values: list[list[float]],
                     title: str, ylabel: str) -> None:
    options = {
        'patch_artist': True,
        'showfliers': False,
        'medianprops': {'color': '#222222', 'linewidth': 1.2},
    }
    labels = [ARM_LABELS[arm] for arm in TREATED_ARMS]
    try:
        boxes = ax.boxplot(values, tick_labels=labels, **options)
    except TypeError as error:
        if 'tick_labels' not in str(error):
            raise
        # Matplotlib <3.9 used ``labels``; newer releases use ``tick_labels``.
        boxes = ax.boxplot(values, labels=labels, **options)
    for patch, arm in zip(boxes['boxes'], TREATED_ARMS, strict=True):
        patch.set_facecolor(ARM_COLORS[arm])
        patch.set_alpha(.85)
    ax.axhline(0., color='#777777', linewidth=.8)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis='x', rotation=45)


def figure_3_response_landscape(contrasts: list[dict[str, str]],
                                font_family: str) -> plt.Figure:
    _require(len(contrasts) == 229376,
             'Figure 3 requires the complete JT1 contrast table.')
    defined = _defined_contrasts(contrasts)
    with plt.rc_context(style(font_family)):
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        primary_values = {
            arm: [float(row['percent_change_from_control']) for row in defined
                  if row['arm'] == arm and row['metric'] == 'AUC_A42'
                  and float(row['checkpoint']) == 2000.]
            for arm in TREATED_ARMS
        }
        c99_values = {
            arm: [float(row['percent_change_from_control']) for row in defined
                  if row['arm'] == arm and row['metric'] == 'C99_max'
                  and float(row['checkpoint']) == 2000.]
            for arm in TREATED_ARMS
        }
        _colored_boxplot(axes[0, 0], [primary_values[arm] for arm in TREATED_ARMS],
                         'Primary response at 2,000 tau', 'Δ AUC Aβ42 (%)')
        panel(axes[0, 0], 'A')
        _colored_boxplot(axes[0, 1], [c99_values[arm] for arm in TREATED_ARMS],
                         'C99 trade-off at 2,000 tau', 'Δ maximum C99 (%)')
        panel(axes[0, 1], 'B')

        paired: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in defined:
            if float(row['checkpoint']) == 2000. and row['metric'] in ('AUC_A42', 'C99_max'):
                paired[(row['unit_id'], row['arm'])][row['metric']] = float(
                    row['percent_change_from_control'])
        for arm in TREATED_ARMS:
            items = [value for (unit, item_arm), value in paired.items()
                     if item_arm == arm and set(value) == {'AUC_A42', 'C99_max'}]
            axes[1, 0].scatter([item['AUC_A42'] for item in items],
                               [item['C99_max'] for item in items], s=8,
                               alpha=.28, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        axes[1, 0].axvline(0., color='#777777', linewidth=.8)
        axes[1, 0].axhline(0., color='#777777', linewidth=.8)
        axes[1, 0].set(xlabel='Δ AUC Aβ42 (%)', ylabel='Δ maximum C99 (%)',
                       title='Mechanistic response–trade-off landscape')
        axes[1, 0].legend(frameon=False, ncol=2, markerscale=2)
        panel(axes[1, 0], 'C')

        for arm in TREATED_ARMS:
            medians = []
            for checkpoint in CHECKPOINTS:
                values = [float(row['percent_change_from_control']) for row in defined
                          if row['arm'] == arm and row['metric'] == 'AUC_A42'
                          and float(row['checkpoint']) == checkpoint]
                medians.append(float(np.median(values)))
            axes[1, 1].plot(CHECKPOINTS, medians, marker='o', linewidth=1.6,
                            color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        axes[1, 1].set(xlabel='Checkpoint (tau)', ylabel='Median Δ AUC Aβ42 (%)',
                       title='Response across the treatment horizon',
                       xticks=CHECKPOINTS)
        axes[1, 1].legend(frameon=False, ncol=1, loc='center left',
                          bbox_to_anchor=(1.01, .5))
        panel(axes[1, 1], 'D')
        figure.suptitle('Therapeutic-response landscape in JT1',
                        fontsize=14, fontweight='bold')
        return figure


def _cell_matrix(contrasts: list[dict[str, str]], arm: str,
                 i_ext_bin: int) -> np.ndarray:
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in contrasts:
        if (row['arm'] == arm and row['metric'] == 'AUC_A42'
                and float(row['checkpoint']) == 2000.
                and row['contrast_reason'] == ''
                and int(row['cell_id'][3]) == i_ext_bin):
            x, y = int(row['cell_id'][1]) - 1, int(row['cell_id'][2]) - 1
            cells[(y, x)].append(float(row['percent_change_from_control']))
    matrix = np.full((4, 4), np.nan)
    for (y, x), values in cells.items():
        matrix[y, x] = float(np.median(values))
    _require(np.all(np.isfinite(matrix)), f'Incomplete JT1 cells for {arm}/bin {i_ext_bin}.')
    return matrix


def figure_4_heterogeneity(contrasts: list[dict[str, str]],
                           font_family: str) -> plt.Figure:
    representative = ('GSI_03', 'GSM_high')
    matrices = {(arm, level): _cell_matrix(contrasts, arm, level)
                for arm in representative for level in range(1, 5)}
    values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    normalizer = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    with plt.rc_context(style(font_family)):
        figure, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
        for row_index, arm in enumerate(representative):
            for column, level in enumerate(range(1, 5)):
                ax = axes[row_index, column]
                image = ax.imshow(matrices[(arm, level)], origin='lower', cmap='viridis',
                                  norm=normalizer)
                ax.set(title=rf'{ARM_LABELS[arm]} · $I_{{ext}}$ bin {level}',
                       xlabel=r'$k_{BACE}$ bin', ylabel=r'$C_i$ bin',
                       xticks=range(4), yticks=range(4))
                ax.set_xticklabels(range(1, 5))
                ax.set_yticklabels(range(1, 5))
                for y in range(4):
                    for x in range(4):
                        ax.text(x, y, f'{matrices[(arm, level)][y, x]:.2f}',
                                ha='center', va='center', fontsize=7,
                                color='white' if normalizer(
                                    matrices[(arm, level)][y, x]) < .35 else '#111111')
        panel(axes[0, 0], 'A')
        panel(axes[1, 0], 'B')
        figure.colorbar(image, ax=axes, label='Median Δ AUC Aβ42 (%)', shrink=.78)
        figure.suptitle('Joint heterogeneity of the primary therapeutic response',
                        fontsize=14, fontweight='bold')
        return figure


def figure_5_surrogate_fidelity(predictions: list[dict[str, str]],
                                metrics: list[dict[str, str]],
                                selections: list[dict[str, str]],
                                font_family: str) -> plt.Figure:
    primary = [row for row in predictions if row['target_id'] == PRIMARY_TARGET]
    _require(len(primary) == 1344 and len(metrics) == 32 and len(selections) == 32,
             'Figure 5 requires complete frozen E1 artifacts.')
    with plt.rc_context(style(font_family)):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8.7), constrained_layout=True)
        for arm in TREATED_ARMS:
            rows = [row for row in primary if row['arm'] == arm]
            truth = [float(row['true_value']) for row in rows]
            predicted = [float(row['prediction']) for row in rows]
            axes[0, 0].scatter(truth, predicted, s=12, alpha=.45,
                               color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        all_values = [float(row[field]) for row in primary
                      for field in ('true_value', 'prediction')]
        limits = min(all_values), max(all_values)
        axes[0, 0].plot(limits, limits, '--', color='#333333', linewidth=1.)
        axes[0, 0].set(xlabel='Mechanistic Δ AUC Aβ42 (%)',
                       ylabel='Surrogate prediction (%)',
                       title='Held-out primary endpoint')
        axes[0, 0].legend(frameon=False, ncol=2)
        panel(axes[0, 0], 'A')

        residuals = [[float(row['prediction']) - float(row['true_value'])
                      for row in primary if row['arm'] == arm]
                     for arm in TREATED_ARMS]
        _colored_boxplot(axes[0, 1], residuals, 'Primary residuals by arm',
                         'Prediction − mechanistic (percentage points)')
        panel(axes[0, 1], 'B')

        matrix = np.full((len(METRICS), len(CHECKPOINTS)), np.nan)
        for row in metrics:
            matrix[METRICS.index(row['metric']), CHECKPOINTS.index(
                float(row['checkpoint']))] = float(row['global_nrmse'])
        _require(np.all(np.isfinite(matrix)) and np.all(matrix > 0.),
                 'The NRMSE heatmap requires 32 finite positive target values.')
        image = axes[0, 2].imshow(
            matrix, aspect='auto', cmap='magma_r',
            norm=LogNorm(vmin=float(np.min(matrix)), vmax=float(np.max(matrix))))
        axes[0, 2].set(title='Global NRMSE across all targets',
                       xticks=range(len(CHECKPOINTS)), yticks=range(len(METRICS)))
        axes[0, 2].set_xticklabels([int(value) for value in CHECKPOINTS])
        axes[0, 2].set_yticklabels([METRIC_LABELS[value] for value in METRICS])
        axes[0, 2].set_xlabel('Checkpoint (tau)')
        figure.colorbar(image, ax=axes[0, 2], label='Global NRMSE (log scale)')
        panel(axes[0, 2], 'C')

        axes[1, 0].scatter([float(row['global_nrmse']) for row in metrics],
                           [float(row['boundary_nrmse']) for row in metrics],
                           color='#4c78a8', s=30)
        maximum = max(float(row['global_nrmse']) for row in metrics)
        axes[1, 0].plot([1e-6, maximum * 1.2], [1e-6, maximum * 1.2], '--',
                        color='#777777')
        axes[1, 0].set(xscale='log', yscale='log', xlabel='Global NRMSE',
                       ylabel='Boundary-cell NRMSE', title='Boundary robustness')
        panel(axes[1, 0], 'D')

        configurations = Counter(row['selected_config_id'] for row in selections)
        names = sorted(configurations)
        bars = axes[1, 1].bar(names, [configurations[name] for name in names],
                              color='#f58518')
        axes[1, 1].bar_label(bars)
        axes[1, 1].set(ylabel='Selected targets', title='Frozen HGB configurations')
        panel(axes[1, 1], 'E')

        selected_by_target = {row['target_id']: row for row in selections}
        axes[1, 2].scatter(
            [float(selected_by_target[row['target_id']]['validation_global_nrmse'])
             for row in metrics],
            [float(row['global_nrmse']) for row in metrics],
            color='#7a5195', s=30)
        validation_max = max(float(row['validation_global_nrmse'])
                             for row in selections)
        test_max = max(float(row['global_nrmse']) for row in metrics)
        upper = max(validation_max, test_max) * 1.2
        axes[1, 2].plot([1e-6, upper], [1e-6, upper], '--', color='#777777')
        axes[1, 2].set(xscale='log', yscale='log',
                       xlabel='Validation global NRMSE',
                       ylabel='Test global NRMSE',
                       title='Validation-to-test consistency')
        panel(axes[1, 2], 'F')
        figure.suptitle('Surrogate fidelity in the single-opening internal test',
                        fontsize=14, fontweight='bold')
        return figure


def figure_6_domain_speed(domain: Mapping[str, object],
                          rejections: Mapping[str, object],
                          performance: Mapping[str, object],
                          font_family: str) -> plt.Figure:
    ranges = domain['factor_ranges']
    benchmark = performance['surrogate_benchmark']
    mechanistic = performance['mechanistic_archive']
    with plt.rc_context(style(font_family)):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        names = [r'$k_{BACE}$ multiplier', r'$C_i$', r'$I_{ext}$']
        keys = ['k_BACE_multiplier', 'C_i', 'I_ext']
        for index, (name, key) in enumerate(zip(names, keys, strict=True)):
            low, high = (float(value) for value in ranges[key])
            axes[0, 0].plot([low, high], [index, index], linewidth=8,
                            solid_capstyle='round', color='#4c78a8')
            axes[0, 0].text(high + .03, index, f'[{low:g}, {high:g}]', va='center')
        axes[0, 0].set(yticks=range(3), yticklabels=names,
                       xlabel='Allowed joint-factor interval',
                       title='Frozen JT1 applicability domain')
        axes[0, 0].text(.98, .43,
                        'All three intervals must hold simultaneously;\nP12 varies one axis at a time.',
                        transform=axes[0, 0].transAxes, ha='right', va='center',
                        bbox={'boxstyle': 'round', 'facecolor': 'white',
                              'edgecolor': '#777777'})
        panel(axes[0, 0], 'A')

        coverage = [100., 100. * int(rejections['in_domain_rows'])
                    / int(rejections['rows'])]
        bars = axes[0, 1].bar(['JT1 internal test', 'P12 external design'], coverage,
                              color=['#54a24b', '#e45756'])
        axes[0, 1].bar_label(bars, fmt='%.0f%%')
        axes[0, 1].set(ylabel='Requests inside domain (%)', ylim=(0, 108),
                       title='Applicability coverage')
        panel(axes[0, 1], 'B')

        rejection_map = rejections['rejections']
        rejection_names = [name.replace('factor_outside_domain:', '')
                           for name in rejection_map]
        rejection_values = [int(rejection_map[name]) for name in rejection_map]
        bars = axes[1, 0].bar(rejection_names, rejection_values,
                              color=['#f58518', '#b279a2'][:len(rejection_names)])
        axes[1, 0].bar_label(bars)
        axes[1, 0].set(ylabel='P12 rows', title='First recorded rejection reason')
        axes[1, 0].text(.98, .95, '0 predictions issued', transform=axes[1, 0].transAxes,
                        ha='right', va='top', fontweight='bold')
        panel(axes[1, 0], 'C')

        labels = ['Batched surrogate\nper request', 'Individual surrogate\nper request',
                  'Archived mechanistic\nper integration']
        times = [float(benchmark['batch_seconds_per_32_target_request']),
                 float(benchmark['individual_seconds_per_32_target_request_median']),
                 float(mechanistic['wall_seconds_per_archived_integration'])]
        bars = axes[1, 1].bar(labels, times,
                              color=['#4c78a8', '#72b7b2', '#777777'])
        axes[1, 1].set(yscale='log', ylabel='Wall time (s, log scale)',
                       title='Operational inference speed')
        axes[1, 1].bar_label(bars, labels=[f'{value:.4g} s' for value in times],
                                  padding=3)
        axes[1, 1].text(.98, .08,
                        f'{float(benchmark["operational_speedup_vs_archived_integration_batch"]):,.0f}× '
                        'operational speedup', transform=axes[1, 1].transAxes,
                        ha='right', fontweight='bold')
        panel(axes[1, 1], 'D')
        figure.suptitle('Applicability guard and operational acceleration',
                        fontsize=14, fontweight='bold')
        return figure
