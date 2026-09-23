"""Extended, read-only statistics for SeqTrack compute dose-response sweeps."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from tracking.analyze_sv_test_results import calc_center_error, calc_norm_center_error
except ModuleNotFoundError:
    from analyze_sv_test_results import calc_center_error, calc_norm_center_error


DOSE_STARTS = (6, 7, 8, 9, 10)
MAIN_STARTS = (6, 8, 9, 10)
COORD_PAIRS = ((6, 8), (6, 9), (6, 10), (6, 12),
               (8, 10), (8, 12), (9, 12), (10, 12))
PROXY_FIELDS = ('conf_mean', 'conf_min', 'raw_conf_mean', 'raw_conf_min',
                'raw_entropy_mean', 'raw_entropy_max', 'raw_margin_mean', 'raw_margin_min',
                'search_cos_mean', 'search_cos_median')


def save_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fraction(values, predicate):
    return float(np.mean(predicate(values))) if len(values) else float('nan')


def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {'frames': len(x), 'pearson_r': float('nan'), 'pearson_p': float('nan'),
                'spearman_rho': float('nan'), 'spearman_p': float('nan')}
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {'frames': len(x), 'pearson_r': float(pearson.statistic),
            'pearson_p': float(pearson.pvalue), 'spearman_rho': float(spearman.statistic),
            'spearman_p': float(spearman.pvalue)}


def matrix_csv(output_dir, dataset, depths, name, matrix):
    columns = ['dataset', 'from_depth'] + [f'to_{depth}' for depth in depths]
    rows = []
    for i, depth in enumerate(depths):
        row = {'dataset': dataset, 'from_depth': int(depth)}
        row.update({f'to_{target}': float(matrix[i, j]) if j > i else ''
                    for j, target in enumerate(depths)})
        rows.append(row)
    save_csv(output_dir / name, rows, columns)


def matrix_plot(ax, matrix, depths, title, cmap='RdBu_r', vmin=None, vmax=None):
    masked = np.ma.masked_invalid(matrix)
    if cmap == 'RdBu_r' and vmin is None and vmax is None:
        limit = float(np.nanmax(abs(matrix)))
        vmin, vmax = -max(limit, 1e-12), max(limit, 1e-12)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_xticks(range(len(depths)), depths)
    ax.set_yticks(range(len(depths)), depths)
    ax.set(xlabel='目标深度', ylabel='当前深度', title=title)
    return image


def run_extended_analysis(output_dir, dataset, depths, keys, y, conf, raw, similarity,
                          gt, pred, oracle, depth_rows, random_seed=42):
    """Write all-pair and sequence-level diagnostics. Inputs are aligned by frame."""
    output_dir = Path(output_dir)
    depths = np.asarray(depths, dtype=int)
    n, count = y.shape
    index = {int(depth): i for i, depth in enumerate(depths)}
    sequences = np.asarray([key[1] for key in keys])
    full = y[:, index[12]]
    main_uplift = {depth: full - y[:, index[depth]] for depth in MAIN_STARTS}

    # Every frame's complete upper-triangle dose-response profile, streamed to disk.
    dose_columns = ['dataset', 'sequence', 'frame'] + [f'iou_{d}' for d in depths]
    dose_columns += [f'u_{a}_{b}' for a in depths for b in depths if a < b]
    with (output_dir / 'frame_compute_response.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dose_columns)
        writer.writeheader()
        for i, key in enumerate(keys):
            row = {'dataset': dataset, 'sequence': key[1], 'frame': key[2]}
            row.update({f'iou_{d}': y[i, index[d]] for d in depths})
            row.update({f'u_{a}_{b}': y[i, index[b]] - y[i, index[a]]
                        for a in depths for b in depths if a < b})
            writer.writerow(row)

    definitions = {
        'mean': lambda a: np.mean(a), 'median': lambda a: np.median(a),
        'p_gt_000': lambda a: np.mean(a > 0),
        'p_gt_002': lambda a: np.mean(a > 0.02),
        'p_gt_005': lambda a: np.mean(a > 0.05),
        'p_gt_010': lambda a: np.mean(a > 0.10),
        'p_negative': lambda a: np.mean(a < 0),
        'p_strong_negative': lambda a: np.mean(a < -0.05),
        'p_near_zero': lambda a: np.mean(abs(a) < 0.01),
    }
    matrices = {name: np.full((count, count), np.nan) for name in definitions}
    for i in range(count):
        for j in range(i+1, count):
            delta = y[:, j] - y[:, i]
            for name, function in definitions.items():
                matrices[name][i, j] = function(delta)
    for name, matrix in matrices.items():
        matrix_csv(output_dir, dataset, depths, f'uplift_matrix_{name}.csv', matrix)

    # Exact project definition of normalized center error is imported above.
    center = np.column_stack([calc_center_error(pred[:, i, :], gt) for i in range(count)])
    norm_center = np.column_stack([calc_norm_center_error(pred[:, i, :], gt) for i in range(count)])
    width = abs(pred[:, :, 2] - gt[:, None, 2]) / gt[:, None, 2]
    height = abs(pred[:, :, 3] - gt[:, None, 3]) / gt[:, None, 3]
    with (output_dir / 'frame_coordinate_errors.csv').open('w', newline='') as file:
        columns = ['dataset', 'sequence', 'frame', 'depth', 'center_error',
                   'normalized_center_error', 'width_relative_error', 'height_relative_error']
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i, key in enumerate(keys):
            for depth in depths:
                j = index[depth]
                writer.writerow({'dataset': dataset, 'sequence': key[1], 'frame': key[2],
                                 'depth': depth, 'center_error': center[i, j],
                                 'normalized_center_error': norm_center[i, j],
                                 'width_relative_error': width[i, j],
                                 'height_relative_error': height[i, j]})
    coordinate_rows = []
    for a, b in COORD_PAIRS:
        improvements = (center[:, index[a]] - center[:, index[b]],
                        norm_center[:, index[a]] - norm_center[:, index[b]],
                        width[:, index[a]] - width[:, index[b]],
                        height[:, index[a]] - height[:, index[b]])
        coordinate_rows.append({'dataset': dataset, 'from_depth': a, 'to_depth': b,
                                'frames': n, 'mean_iou_uplift': float(np.mean(y[:, index[b]] - y[:, index[a]])),
                                'mean_center_improvement': float(np.mean(improvements[0])),
                                'mean_normalized_center_improvement': float(np.mean(improvements[1])),
                                'mean_width_error_improvement': float(np.mean(improvements[2])),
                                'mean_height_error_improvement': float(np.mean(improvements[3])),
                                'p_center_improves': fraction(improvements[0], lambda v: v > 0),
                                'p_width_improves': fraction(improvements[2], lambda v: v > 0),
                                'p_height_improves': fraction(improvements[3], lambda v: v > 0)})
    save_csv(output_dir / 'coordinate_uplift_decomposition.csv', coordinate_rows)

    # Three intervals expose shallow and late-stage non-monotonicity separately.
    nonmono_rows = []
    with (output_dir / 'frame_nonmonotonicity.csv').open('w', newline='') as file:
        columns = ['dataset', 'sequence', 'frame', 'start_depth', 'end_depth',
                   'negative_step_count', 'strong_negative_step_count', 'max_positive_step',
                   'max_negative_step', 'depth_of_max_positive_step', 'depth_of_max_negative_step',
                   'is_monotonic_non_decreasing', 'has_strong_nonmonotonicity']
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for start in (1, 6, 8):
            step = np.diff(y[:, index[start]:index[12]+1], axis=1)
            negative = (step < 0).sum(axis=1)
            strong = (step < -0.05).sum(axis=1)
            minimum, maximum = step.min(axis=1), step.max(axis=1)
            for i, key in enumerate(keys):
                writer.writerow({'dataset': dataset, 'sequence': key[1], 'frame': key[2],
                                 'start_depth': start, 'end_depth': 12,
                                 'negative_step_count': int(negative[i]),
                                 'strong_negative_step_count': int(strong[i]),
                                 'max_positive_step': max(0.0, float(maximum[i])),
                                 'max_negative_step': min(0.0, float(minimum[i])),
                                 'depth_of_max_positive_step': start + int(np.argmax(step[i])) if maximum[i] > 0 else '',
                                 'depth_of_max_negative_step': start + int(np.argmin(step[i])) if minimum[i] < 0 else '',
                                 'is_monotonic_non_decreasing': bool(negative[i] == 0),
                                 'has_strong_nonmonotonicity': bool(strong[i] > 0)})
            nonmono_rows.append({'dataset': dataset, 'start_depth': start, 'end_depth': 12,
                                 'frames': n, 'p_monotonic': float(np.mean(negative == 0)),
                                 'p_nonmonotonic': float(np.mean(negative > 0)),
                                 'p_strong_nonmonotonic': float(np.mean(strong > 0)),
                                 'mean_negative_steps': float(np.mean(negative)),
                                 'median_negative_steps': float(np.median(negative))})
    save_csv(output_dir / 'nonmonotonicity_summary.csv', nonmono_rows)

    delayed_rows = []
    delayed_by_start = {}
    for start in DOSE_STARTS:
        local = y[:, index[start+1]] - y[:, index[start]]
        future = full - y[:, index[start]]
        delayed_by_start[start] = (local, future)
        delayed_rows.append({'dataset': dataset, 'start_depth': start, 'frames': n,
                             'p_abs_local_lt_0.01_future_gt_0.05':
                                 fraction(local, lambda a: (abs(a) < 0.01) & (future > 0.05)),
                             'p_local_negative_future_gt_0.05':
                                 fraction(local, lambda a: (a < 0) & (future > 0.05)),
                             'p_local_le_0.02_future_gt_0.10':
                                 fraction(local, lambda a: (a <= 0.02) & (future > 0.10)),
                             'p_local_negative_future_gt_0.10':
                                 fraction(local, lambda a: (a < 0) & (future > 0.10))})
    save_csv(output_dir / 'delayed_gain_summary.csv', delayed_rows)

    # Threshold sensitivity is descriptive: fixed grid, no optimization.
    sensitivity = []
    for start in MAIN_STARTS:
        current, delta = y[:, index[start]], main_uplift[start]
        for difficulty_threshold in (0.3, 0.5, 0.7):
            hard = current < difficulty_threshold
            for uplift_threshold in (0.02, 0.05, 0.10):
                high = delta > uplift_threshold
                sensitivity.append({'dataset': dataset, 'from_depth': start, 'to_depth': 12,
                                    'difficulty_threshold': difficulty_threshold,
                                    'uplift_threshold': uplift_threshold, 'frames': n,
                                    'n_hard': int(hard.sum()), 'n_easy': int((~hard).sum()),
                                    'p_high_uplift_given_hard': float(high[hard].mean()) if hard.any() else float('nan'),
                                    'p_high_uplift_given_easy': float(high[~hard].mean()) if (~hard).any() else float('nan'),
                                    'p_hard_and_high_uplift': float(np.mean(hard & high)),
                                    'p_hard_and_low_uplift': float(np.mean(hard & ~high)),
                                    'p_easy_and_high_uplift': float(np.mean(~hard & high)),
                                    'p_easy_and_low_uplift': float(np.mean(~hard & ~high)),
                                    'difficulty_mismatch_rate': float(np.mean((hard & ~high) | (~hard & high)))})
    save_csv(output_dir / 'difficulty_uplift_threshold_sensitivity.csv', sensitivity)

    recoverability = []
    for start in MAIN_STARTS:
        shallow_success = y[:, index[start]] >= 0.5
        full_success = full >= 0.5
        delta = main_uplift[start]
        categories = {'A_both_success': shallow_success & full_success,
                      'B_recoverable': ~shallow_success & full_success,
                      'C_both_fail': ~shallow_success & ~full_success,
                      'D_deeper_failure': shallow_success & ~full_success}
        row = {'dataset': dataset, 'start_depth': start, 'full_depth': 12, 'frames': n}
        for name, mask in categories.items():
            row[f'n_{name}'] = int(mask.sum())
            row[f'p_{name}'] = float(mask.mean())
            if name.startswith(('B_', 'C_', 'D_')):
                row[f'mean_uplift_{name}'] = float(np.mean(delta[mask])) if mask.any() else float('nan')
                row[f'median_uplift_{name}'] = float(np.median(delta[mask])) if mask.any() else float('nan')
        recoverability.append(row)
    save_csv(output_dir / 'recoverability_summary.csv', recoverability)

    best_index = np.argmax(y, axis=1)
    best_depth = depths[best_index]
    best_iou = y[np.arange(n), best_index]
    gain = best_iou - full
    best_rows = [{'dataset': dataset, 'best_depth': int(depth), 'frames': n,
                  'n_best_depth': int(np.sum(best_depth == depth)),
                  'p_best_depth': float(np.mean(best_depth == depth)),
                  'mean_gain_over_depth12': float(np.mean(gain)),
                  'p_gain_gt_0.02': fraction(gain, lambda a: a > 0.02),
                  'p_gain_gt_0.05': fraction(gain, lambda a: a > 0.05),
                  'p_gain_gt_0.10': fraction(gain, lambda a: a > 0.10)}
                 for depth in depths]
    save_csv(output_dir / 'best_depth_summary.csv', best_rows)

    proxies = {**conf, **raw, **similarity}
    proxy_rows = []
    for start in MAIN_STARTS:
        for target, delta in ((start+1, y[:, index[start+1]] - y[:, index[start]]),
                              (12, main_uplift[start])):
            for field in PROXY_FIELDS:
                proxy_rows.append({'dataset': dataset, 'from_depth': start, 'to_depth': target,
                                   'proxy': field, **correlation(proxies[field][:, index[start]], delta),
                                   'note': 'pre-window raw statistics' if field.startswith('raw_') else
                                           'search token cosine with preceding layer' if field.startswith('search_') else
                                           'confidence may be affected by Hann window'})
    save_csv(output_dir / 'proxy_uplift_correlation.csv', proxy_rows)

    cost_rows = [{'dataset': dataset, 'depth': int(depth), 'encoder_blocks': int(depth),
                  'cost_proxy': int(depth), 'encoder_macs': 'unavailable',
                  'decoder_macs': 'unavailable', 'full_pipeline_macs': 'unavailable',
                  'latency': 'unavailable'} for depth in depths]
    save_csv(output_dir / 'cost_profile.csv', cost_rows)
    adjusted = []
    for i, start in enumerate(depths):
        for target in depths[i+1:]:
            values = (y[:, index[target]] - y[:, index[start]]) / (target - start)
            adjusted.append({'dataset': dataset, 'from_depth': int(start), 'to_depth': int(target),
                             'delta_encoder_blocks': int(target - start), 'frames': n,
                             'mean_cost_adjusted_uplift': float(np.mean(values)),
                             'median_cost_adjusted_uplift': float(np.median(values)),
                             'p_positive': fraction(values, lambda a: a > 0)})
    save_csv(output_dir / 'cost_adjusted_uplift_summary.csv', adjusted)

    # Sequence summaries, with success-conditioned oracle to expose full-model failures.
    sequence_rows = []
    for sequence in sorted(set(sequences)):
        mask = sequences == sequence
        chosen = oracle[0.02][mask]
        frame_count = int(mask.sum())
        row = {'dataset': dataset, 'sequence': sequence, 'frames': frame_count}
        for depth in (6, 8, 9, 10, 12):
            row[f'mean_iou_{depth}'] = float(np.mean(y[mask, index[depth]]))
        for target in (8, 9, 10, 12):
            row[f'mean_uplift_6_to_{target}'] = float(np.mean(y[mask, index[target]] - y[mask, index[6]]))
        for start in (6, 8, 10):
            values = main_uplift[start][mask]
            row[f'mean_uplift_{start}_to_12'] = float(np.mean(values))
            row[f'median_uplift_{start}_to_12'] = float(np.median(values))
            row[f'p_uplift_{start}_to_12_gt_0.05'] = float(np.mean(values > 0.05))
        hard = y[mask, index[6]] < 0.5
        high = main_uplift[6][mask] > 0.05
        row['difficulty_mismatch_rate_6_to_12'] = float(np.mean((hard & ~high) | (~hard & high)))
        row['oracle_mean_depth_epsilon_0.02'] = float(np.mean(chosen))
        row['oracle_median_depth_epsilon_0.02'] = float(np.median(chosen))
        success = mask & (full >= 0.5)
        row['full_success_frames'] = int(success.sum())
        row['oracle_mean_depth_full_success_epsilon_0.02'] = (
            float(np.mean(oracle[0.02][success])) if success.any() else float('nan'))
        row['p_nonmonotonic_6_to_12'] = float(np.mean((np.diff(y[mask, index[6]:], axis=1) < 0).any(axis=1)))
        local, future = delayed_by_start[6]
        row['p_delayed_gain_6'] = float(np.mean((abs(local[mask]) < 0.01) & (future[mask] > 0.05)))
        sequence_rows.append(row)
    save_csv(output_dir / 'sequence_uplift_summary.csv', sequence_rows)

    # Cluster bootstrap: resample complete sequences, then aggregate all their frames.
    unique_sequences = sorted(set(sequences))
    seq_masks = [sequences == sequence for sequence in unique_sequences]
    metrics = {}
    for depth in depths:
        metrics[f'mean_iou_{depth}'] = (y[:, index[depth]], np.ones(n, dtype=bool))
    for start in (6, 8, 10):
        metrics[f'mean_uplift_{start}_to_12'] = (main_uplift[start], np.ones(n, dtype=bool))
    for start in (6, 8):
        metrics[f'p_uplift_{start}_to_12_gt_0.05'] = ((main_uplift[start] > 0.05).astype(float),
                                                      np.ones(n, dtype=bool))
    hard = y[:, index[6]] < 0.5
    high = main_uplift[6] > 0.05
    metrics['difficulty_mismatch_rate_6_to_12'] = (((hard & ~high) | (~hard & high)).astype(float),
                                                    np.ones(n, dtype=bool))
    metrics['oracle_mean_depth_full_success_epsilon_0.02'] = (oracle[0.02].astype(float), full >= 0.5)
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(unique_sequences), size=(1000, len(unique_sequences)))
    bootstrap_rows = []
    for name, (values, eligible) in metrics.items():
        sums = np.array([np.sum(values[mask & eligible]) for mask in seq_masks], dtype=float)
        counts = np.array([np.sum(mask & eligible) for mask in seq_masks], dtype=float)
        if counts.sum() == 0:
            bootstrap_rows.append({'dataset': dataset, 'metric': name, 'sequences': len(unique_sequences),
                                   'bootstrap_replicates': 0, 'seed': random_seed,
                                   'point_estimate': float('nan'), 'bootstrap_mean': float('nan'),
                                   'ci_95_lower': float('nan'), 'ci_95_upper': float('nan')})
            continue
        estimates = sums[sampled].sum(axis=1) / np.maximum(counts[sampled].sum(axis=1), 1)
        valid = counts[sampled].sum(axis=1) > 0
        estimates = estimates[valid]
        bootstrap_rows.append({'dataset': dataset, 'metric': name, 'sequences': len(unique_sequences),
                               'bootstrap_replicates': len(estimates), 'seed': random_seed,
                               'point_estimate': float(np.sum(sums) / np.sum(counts)),
                               'bootstrap_mean': float(np.mean(estimates)),
                               'ci_95_lower': float(np.quantile(estimates, 0.025)),
                               'ci_95_upper': float(np.quantile(estimates, 0.975))})
    save_csv(output_dir / 'bootstrap_confidence_intervals.csv', bootstrap_rows)

    make_plots(output_dir, dataset, depths, y, matrices, delayed_by_start, oracle,
               best_depth, coordinate_rows, sequence_rows, depth_rows)
    return {'frames': n, 'sequences': len(unique_sequences), 'bootstrap_replicates': 1000,
            'cost_note': 'cost(d)=number of Encoder blocks; latency and MACs unavailable'}


def make_plots(output_dir, dataset, depths, y, matrices, delayed, oracle,
               best_depth, coordinate_rows, sequence_rows, depth_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    cjk_fonts = [path for path in font_manager.findSystemFonts() if 'NotoSansCJK-Regular' in path]
    if cjk_fonts:
        font_manager.fontManager.addfont(cjk_fonts[0])
        plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=cjk_fonts[0]).get_name(),
                                           'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    advisor = output_dir / 'advisor_figures'
    advisor.mkdir(exist_ok=True)
    sources = {}

    def export(fig, filename, csv_files):
        fig.tight_layout()
        fig.savefig(advisor / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        sources[filename] = csv_files

    # Requested matrix plots, plus presentation copies at 300 dpi.
    for metric, filename, title, cmap in (
        ('mean', 'uplift_matrix_mean.png', '平均 Compute Uplift', 'RdBu_r'),
        ('p_gt_005', 'uplift_matrix_p_gt_005.png', 'P(Uplift > 0.05)', 'Blues'),
        ('p_negative', 'uplift_matrix_p_negative.png', 'P(Uplift < 0)', 'Oranges')):
        fig, ax = plt.subplots(figsize=(8, 7))
        image = matrix_plot(ax, matrices[metric], depths, title, cmap=cmap)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=300); plt.close(fig)
    for metric, filename, title, cmap in (
        ('mean', '02_mean_uplift_matrix.png', '平均 Compute Uplift', 'RdBu_r'),
        ('p_gt_005', '03_high_uplift_probability_matrix.png', 'P(Uplift > 0.05)', 'Blues'),
        ('p_negative', '04_negative_uplift_probability_matrix.png', 'P(Uplift < 0)', 'Oranges')):
        fig, ax = plt.subplots(figsize=(8, 7))
        image = matrix_plot(ax, matrices[metric], depths, title, cmap=cmap)
        fig.colorbar(image, ax=ax, shrink=0.8)
        export(fig, filename, [f'uplift_matrix_{metric}.csv'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, [row['mean_iou'] for row in depth_rows], marker='o', label='均值')
    ax.plot(depths, [row['median_iou'] for row in depth_rows], marker='o', label='中位数')
    ax.fill_between(depths, [row['q25_iou'] for row in depth_rows],
                    [row['q75_iou'] for row in depth_rows], alpha=0.2, label='25%-75%')
    ax.set(xlabel='Encoder 深度', ylabel='IoU', title='深度与跟踪结果', xticks=depths)
    ax.legend(); export(fig, '01_depth_vs_iou.png', ['depth_summary.csv'])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, start in zip(axes.flat, MAIN_STARTS):
        current = y[:, np.where(depths == start)[0][0]]
        delta = y[:, -1] - current
        ax.scatter(current, delta, s=3, alpha=0.12)
        ax.axvline(0.5, color='#bb4444', linestyle='--')
        ax.axhline(0.05, color='#bb4444', linestyle='--')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set(title=f'{start} → 12', xlabel=f'IoU@{start}', ylabel='Compute Uplift')
    export(fig, '05_difficulty_vs_uplift.png', ['difficulty_uplift_conditional.csv',
                                                'frame_compute_response.csv'])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, start in zip(axes.flat, DOSE_STARTS):
        local, future = delayed[start]
        ax.scatter(local, future, s=3, alpha=0.12)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.axhline(0.05, color='#bb4444', linestyle='--')
        ax.set(title=f'起点 {start}', xlabel='下一层收益', ylabel='到 depth 12 的收益')
    axes.flat[-1].axis('off')
    export(fig, '06_local_vs_future_gain.png', ['delayed_gain_summary.csv', 'frame_compute_response.csv'])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, start in zip(axes.flat, DOSE_STARTS):
        local, future = delayed[start]
        ax.scatter(local, future, s=3, alpha=0.12)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set(title=f'{start} → {start+1} / 12', xlabel='Local gain', ylabel='Future gain')
    axes.flat[-1].axis('off')
    fig.tight_layout(); fig.savefig(output_dir / 'local_gain_vs_future_gain.png', dpi=160); plt.close(fig)

    success = y[:, -1] >= 0.5
    chosen = oracle[0.02][success]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(depths, [np.mean(chosen == d) if len(chosen) else 0 for d in depths], color='#4477aa')
    ax.set(xlabel='Oracle 深度', ylabel='成功帧比例', title='Full-depth 成功帧的 Oracle 深度', xticks=depths)
    export(fig, '07_oracle_depth_distribution_success_frames.png', ['oracle_summary_stratified.csv'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(depths, [np.mean(best_depth == d) for d in depths], color='#4477aa')
    ax.set(xlabel='Best depth', ylabel='帧比例', title='逐帧最优深度分布', xticks=depths)
    export(fig, '08_best_depth_distribution.png', ['best_depth_summary.csv'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(depths, [np.mean(best_depth == d) for d in depths], color='#4477aa')
    ax.set(xlabel='Best depth', ylabel='Frame proportion', xticks=depths)
    fig.tight_layout(); fig.savefig(output_dir / 'best_depth_distribution.png', dpi=160); plt.close(fig)

    labels = [f"{row['from_depth']}→{row['to_depth']}" for row in coordinate_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels)); width = 0.25
    for j, (field, label) in enumerate((('p_center_improves', '中心定位改善'),
                                        ('p_width_improves', '宽度改善'),
                                        ('p_height_improves', '高度改善'))):
        ax.bar(x + (j-1)*width, [row[field] for row in coordinate_rows], width, label=label)
    ax.set_xticks(x, labels); ax.set(ylabel='改善帧比例', title='坐标级收益分解')
    ax.legend(); export(fig, '09_coordinate_uplift_decomposition.png',
                        ['coordinate_uplift_decomposition.csv'])

    ordered = sorted(sequence_rows, key=lambda row: row['mean_uplift_6_to_12'])
    uplift_pairs = ((6, 8), (6, 9), (6, 10), (6, 12), (8, 12), (10, 12))
    values = np.array([[row[f'mean_uplift_{a}_to_{b}'] for a, b in uplift_pairs] for row in ordered])
    limit = max(float(np.max(abs(values))), 1e-12)
    fig, ax = plt.subplots(figsize=(8, max(6, min(24, len(ordered)*0.15))))
    image = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(uplift_pairs)), [f'{a}→{b}' for a, b in uplift_pairs])
    tick_step = max(1, len(ordered)//25)
    ticks = list(range(0, len(ordered), tick_step))
    ax.set_yticks(ticks, [ordered[i]['sequence'] for i in ticks], fontsize=7)
    ax.set(xlabel='Encoder 深度区间', ylabel='序列', title='不同序列的平均 Compute Uplift')
    fig.colorbar(image, ax=ax, label='平均 IoU 收益')
    export(fig, '10_sequence_uplift_heatmap.png', ['sequence_uplift_summary.csv'])
    fig, ax = plt.subplots(figsize=(8, max(6, min(24, len(ordered)*0.15))))
    image = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(uplift_pairs)), [f'{a}->{b}' for a, b in uplift_pairs])
    ax.set_yticks(ticks, [ordered[i]['sequence'] for i in ticks], fontsize=7)
    ax.set(xlabel='Encoder depth', ylabel='Sequence')
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(output_dir / 'sequence_uplift_heatmap.png', dpi=160); plt.close(fig)

    with (advisor / 'figure_sources.json').open('w') as file:
        json.dump({'dataset': dataset, 'dpi': 300, 'figures': sources}, file, indent=2)
