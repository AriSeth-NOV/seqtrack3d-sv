"""Analyze open-loop SeqTrack encoder-depth sweeps; never runs the tracker."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


FIXED_PAIRS = [(6, 8), (6, 9), (6, 10), (6, 12), (8, 10), (8, 12), (9, 12), (10, 12)]
SCATTER_PAIRS = [(6, 8), (6, 9), (6, 12), (8, 10), (8, 12), (10, 12)]
EPSILONS = (0.01, 0.02, 0.05)
LAMBDAS = (0.0, 0.001, 0.002, 0.005, 0.01)
CONFIDENCES = ('conf_mean', 'conf_min', 'conf_x', 'conf_y', 'conf_w', 'conf_h')
RAW_COORDINATE_FIELDS = tuple(f'raw_{coordinate}_{metric}' for coordinate in 'xywh'
                              for metric in ('top1', 'top2', 'margin', 'entropy'))
RAW_SUMMARY_FIELDS = ('raw_conf_mean', 'raw_conf_min', 'raw_entropy_mean', 'raw_entropy_max',
                      'raw_margin_mean', 'raw_margin_min', 'sequence_raw_logprob')
SIMILARITY_FIELDS = ('search_cos_mean', 'search_cos_median')
PROXY_FIELDS = ('conf_mean', 'conf_min', 'raw_conf_mean', 'raw_conf_min',
                'raw_entropy_mean', 'raw_entropy_max', 'raw_margin_mean', 'raw_margin_min',
                'search_cos_mean', 'search_cos_median')
CONFIDENCE_NOTE = 'current confidence is window-affected when Hann window is enabled'
GROUPS = (
    ('A_full_eq_0', lambda y: y == 0),
    ('B_full_0_to_0.3', lambda y: (0 < y) & (y < 0.3)),
    ('C_full_0.3_to_0.5', lambda y: (0.3 <= y) & (y < 0.5)),
    ('D_full_0.5_to_0.7', lambda y: (0.5 <= y) & (y < 0.7)),
    ('E_full_ge_0.7', lambda y: y >= 0.7),
    ('full_success_ge_0.5', lambda y: y >= 0.5),
)
STRATA = (
    ('all', lambda y: np.ones_like(y, dtype=bool)),
    ('full_gt_0', lambda y: y > 0),
    ('full_ge_0.3', lambda y: y >= 0.3),
    ('full_ge_0.5', lambda y: y >= 0.5),
    ('full_ge_0.7', lambda y: y >= 0.7),
)


def summarize(values):
    if len(values) == 0:
        return dict(mean=float('nan'), median=float('nan'), std=float('nan'),
                    q10=float('nan'), q25=float('nan'), q75=float('nan'), q90=float('nan'))
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                std=float(np.std(values)), q10=float(q10), q25=float(q25),
                q75=float(q75), q90=float(q90))


def fraction(values, predicate):
    return float(np.mean(predicate(values))) if len(values) else float('nan')


def save_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_inputs(inputs, dataset):
    paths = sorted(set(p for root in inputs for p in
                       (root.rglob('encoder_sweep.csv') if root.is_dir() else [root])))
    if not paths:
        raise ValueError('No encoder_sweep.csv files found')
    frames = defaultdict(dict)
    bad_rows = []
    duplicate_rows = []
    row_count = 0
    legacy_dataset_rows = 0
    for path in paths:
        with path.open(newline='') as file:
            reader = csv.DictReader(file)
            required = {'dataset', 'sequence', 'frame', 'depth', 'iou', *CONFIDENCES,
                        *RAW_COORDINATE_FIELDS, *RAW_SUMMARY_FIELDS, *SIMILARITY_FIELDS,
                        'token_x', 'token_y', 'token_w', 'token_h',
                        'gt_x', 'gt_y', 'gt_w', 'gt_h',
                        'pred_x', 'pred_y', 'pred_w', 'pred_h'}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f'{path} lacks required fields: {sorted(required - set(reader.fieldnames or []))}')
            for line, row in enumerate(reader, start=2):
                row_count += 1
                if not row.get('dataset'):
                    legacy_dataset_rows += 1
                try:
                    source_dataset = row.get('dataset') or dataset  # legacy sweep CSVs did not store dataset
                    if source_dataset != dataset:
                        raise ValueError(f'dataset {source_dataset!r} differs from --dataset {dataset!r}')
                    sequence = row['sequence']
                    frame = int(row['frame'])
                    depth = int(row['depth'])
                    iou = float(row['iou'])
                    box = tuple(float(row[f'pred_{axis}']) for axis in 'xywh')
                    gt = tuple(float(row[f'gt_{axis}']) for axis in 'xywh')
                    tokens = tuple(int(row[f'token_{axis}']) for axis in 'xywh')
                    confidence = {field: float(row[field]) for field in CONFIDENCES}
                    raw = {field: float(row[field]) for field in RAW_COORDINATE_FIELDS + RAW_SUMMARY_FIELDS}
                    similarity = {field: (float(row[field]) if row[field] else float('nan'))
                                  for field in SIMILARITY_FIELDS}
                    numbers = (iou, *box, *gt, *confidence.values(), *raw.values())
                    if not sequence or frame < 1 or depth < 1 or not all(map(math.isfinite, numbers)):
                        raise ValueError('empty sequence, bad index or non-finite value')
                    if not 0 <= iou <= 1:
                        raise ValueError(f'IoU outside [0,1]: {iou}')
                    if gt[2] <= 0 or gt[3] <= 0:
                        raise ValueError('non-positive GT width or height')
                    if any(token < 0 for token in tokens):
                        raise ValueError('negative coordinate token')
                    if depth == 1 and any(math.isfinite(v) for v in similarity.values()):
                        raise ValueError('depth 1 has no preceding similarity')
                    if depth > 1 and not all(math.isfinite(v) and -1.00001 <= v <= 1.00001
                                             for v in similarity.values()):
                        raise ValueError('missing or invalid search cosine similarity')
                    if any(not (0 <= raw[f'raw_{axis}_top2'] <= raw[f'raw_{axis}_top1'] <= 1)
                           or raw[f'raw_{axis}_margin'] < -1e-7 or raw[f'raw_{axis}_entropy'] < 0
                           for axis in 'xywh'):
                        raise ValueError('invalid raw probability, margin or entropy')
                except (ValueError, TypeError, KeyError) as error:
                    bad_rows.append({'file': str(path), 'line': line, 'reason': str(error)})
                    continue
                key = (source_dataset, sequence, frame)
                if depth in frames[key]:
                    duplicate_rows.append({'file': str(path), 'line': line, 'key': key, 'depth': depth})
                    continue
                frames[key][depth] = {'iou': iou, 'box': box, 'gt': gt,
                                      'confidence': confidence, 'raw': raw, 'similarity': similarity}
    return paths, frames, row_count, bad_rows, duplicate_rows, legacy_dataset_rows


def compare_boxes(frames, results_dir, full_depth, tolerance):
    rows = []
    for sequence in sorted({key[1] for key in frames}):
        path = results_dir / f'{sequence}.txt'
        if not path.is_file():
            rows.append({'sequence': sequence, 'status': 'missing_file', 'compared_frames': 0,
                         'max_abs_box_error': ''})
            continue
        with path.open() as file:
            boxes = [[float(v) for v in line.split()] for line in file if line.strip()]
        selected = [(key[2], value) for key, value in frames.items() if key[1] == sequence]
        errors = []
        bad = len(boxes) != max(frame for frame, _ in selected) + 1
        for frame, value in selected:
            if frame >= len(boxes) or len(boxes[frame]) != 4 or full_depth not in value:
                bad = True
                continue
            errors.append(max(abs(a-b) for a, b in zip(boxes[frame], value[full_depth]['box'])))
        maximum = max(errors) if errors else float('nan')
        status = 'missing_frames' if bad else 'match' if maximum <= tolerance else 'mismatch'
        rows.append({'sequence': sequence, 'status': status, 'compared_frames': len(errors),
                     'max_abs_box_error': maximum})
    return rows


def quality_gate(args, paths, frames, row_count, bad_rows, duplicate_rows, legacy_dataset_rows, depths):
    observed = {key[1] for key in frames}
    expected = ({path.name for path in args.dataset_root.iterdir() if path.is_dir()}
                if args.expected_sequences.lower() == 'all'
                else set(args.expected_sequences.split(',')))
    missing_sequences = sorted(expected - observed)
    unexpected_sequences = sorted(observed - expected)
    incomplete_frames = [key for key, values in frames.items() if sorted(values) != depths]
    inconsistent_gt = [key for key, values in frames.items()
                       if len({value['gt'] for value in values.values()}) != 1]
    coverage = []
    if args.dataset_root:
        for sequence in sorted(observed):
            annotation = args.dataset_root / sequence / 'groundTruth.rect'
            if not annotation.is_file():
                coverage.append({'sequence': sequence, 'status': 'missing_annotation'})
                continue
            with annotation.open() as file:
                count = sum(bool(line.strip()) for line in file)
            actual = {key[2] for key in frames if key[1] == sequence}
            missing = sorted(set(range(1, count)) - actual)
            extra = sorted(actual - set(range(1, count)))
            coverage.append({'sequence': sequence, 'status': 'complete' if not missing and not extra else 'incomplete',
                             'expected_frames': count - 1, 'observed_frames': len(actual),
                             'missing_count': len(missing), 'extra_count': len(extra),
                             'missing_examples': missing[:20], 'extra_examples': extra[:20]})
    sweep_comparison = compare_boxes(frames, args.sweep_bbox_results_dir, args.full_depth, args.box_tolerance) if args.sweep_bbox_results_dir else []
    baseline_comparison = compare_boxes(frames, args.baseline_results_dir, args.full_depth, args.box_tolerance) if args.baseline_results_dir else []
    if sweep_comparison:
        save_csv(args.output_dir / 'sweep_trajectory_comparison.csv',
                 [{'dataset': args.dataset, **row} for row in sweep_comparison])
    if baseline_comparison:
        save_csv(args.output_dir / 'baseline_comparison.csv',
                 [{'dataset': args.dataset, **row} for row in baseline_comparison])
    errors = []
    if bad_rows: errors.append(f'{len(bad_rows)} invalid rows')
    if duplicate_rows: errors.append(f'{len(duplicate_rows)} duplicate rows')
    if incomplete_frames: errors.append(f'{len(incomplete_frames)} incomplete frames')
    if inconsistent_gt: errors.append(f'{len(inconsistent_gt)} frames have inconsistent GT across depths')
    if missing_sequences: errors.append(f'missing sequences: {missing_sequences}')
    if unexpected_sequences: errors.append(f'unexpected sequences: {unexpected_sequences}')
    if any(row['status'] != 'complete' for row in coverage): errors.append('incomplete sequence coverage')
    if sweep_comparison and any(row['status'] != 'match' for row in sweep_comparison):
        errors.append('sweep trajectory mismatch')
    if baseline_comparison and any(row['status'] != 'match' for row in baseline_comparison):
        errors.append('baseline mismatch')
    if depths != list(range(1, args.full_depth + 1)):
        errors.append(f'depths {depths} are not 1..{args.full_depth}')
    quality = {'dataset': args.dataset, 'csv_files': len(paths), 'sequences': len(observed),
               'frames': len(frames), 'rows': row_count, 'invalid_rows': len(bad_rows),
               'legacy_rows_without_dataset': legacy_dataset_rows,
               'duplicate_rows': len(duplicate_rows), 'incomplete_frames': len(incomplete_frames),
               'inconsistent_gt_frames': len(inconsistent_gt),
               'missing_sequences': missing_sequences, 'unexpected_sequences': unexpected_sequences,
               'sequence_coverage': coverage, 'sweep_trajectory': sweep_comparison,
               'baseline': baseline_comparison,
               'max_abs_box_error': max([float(row['max_abs_box_error']) for row in
                                         sweep_comparison + baseline_comparison if row['max_abs_box_error'] != ''],
                                        default=None),
               'invalid_examples': bad_rows[:20], 'duplicate_examples': duplicate_rows[:20],
               'incomplete_examples': incomplete_frames[:20], 'errors': errors,
               'confidence_note': CONFIDENCE_NOTE}
    with (args.output_dir / 'data_quality.json').open('w') as file:
        json.dump(quality, file, indent=2)
    if errors:
        raise ValueError('Data quality gate failed: ' + '; '.join(errors) +
                         f'. See {args.output_dir / "data_quality.json"}')
    return quality


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


def oracle_depths(y, depths, full_index, epsilon):
    eligible = y >= (y[:, [full_index]] - epsilon)
    return depths[np.argmax(eligible, axis=1)]  # full depth is always eligible


def oracle_summary(dataset, label, epsilon, selected):
    row = {'dataset': dataset, 'group': label, 'epsilon': epsilon, 'frames': len(selected),
           'mean_depth': float(np.mean(selected)) if len(selected) else float('nan'),
           'median_depth': float(np.median(selected)) if len(selected) else float('nan')}
    for threshold in (6, 8, 9, 10, 11):
        row[f'p_depth_le_{threshold}'] = fraction(selected, lambda a, t=threshold: a <= t)
    row['p_depth_eq_12'] = fraction(selected, lambda a: a == 12)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--dataset', default='sv248s', help='Dataset label, e.g. sv248s or viso')
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--expected_sequences', required=True,
                        help='Comma-separated names, or ALL for every sequence directory under --dataset_root')
    parser.add_argument('--sweep_bbox_results_dir', type=Path, required=True)
    parser.add_argument('--baseline_results_dir', type=Path, required=True)
    parser.add_argument('--box_tolerance', type=float, default=1e-4)
    parser.add_argument('--full_depth', type=int, default=12)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, frames, row_count, bad_rows, duplicate_rows, legacy_dataset_rows = parse_inputs(args.inputs, args.dataset)
    depths_list = sorted({depth for value in frames.values() for depth in value})
    quality = quality_gate(args, paths, frames, row_count, bad_rows, duplicate_rows,
                           legacy_dataset_rows, depths_list)
    depths = np.array(depths_list)
    keys = sorted(frames)
    y = np.array([[frames[key][depth]['iou'] for depth in depths] for key in keys], dtype=float)
    conf = {field: np.array([[frames[key][depth]['confidence'][field] for depth in depths] for key in keys])
            for field in CONFIDENCES}
    full_index = depths_list.index(args.full_depth)
    full = y[:, full_index]
    sequences = np.array([key[1] for key in keys])
    dataset = args.dataset
    pairs = list(zip(depths_list[:-1], depths_list[1:])) + FIXED_PAIRS
    pairs = list(dict.fromkeys(pairs))
    index = {depth: i for i, depth in enumerate(depths_list)}
    uplift = {(d1, d2): y[:, index[d2]] - y[:, index[d1]] for d1, d2 in pairs}

    depth_rows = []
    sequence_depth_rows = []
    for depth in depths_list:
        values = y[:, index[depth]]
        s = summarize(values)
        depth_rows.append({'dataset': dataset, 'depth': depth, 'frames': len(values),
                           **{f'{name}_iou': value for name, value in s.items()},
                           'p_iou_eq_0': fraction(values, lambda a: a == 0),
                           'p_iou_lt_0.3': fraction(values, lambda a: a < 0.3),
                           'p_iou_lt_0.5': fraction(values, lambda a: a < 0.5),
                           'p_iou_ge_0.5': fraction(values, lambda a: a >= 0.5),
                           'p_iou_ge_0.7': fraction(values, lambda a: a >= 0.7)})
        for sequence in sorted(set(sequences)):
            sample = values[sequences == sequence]
            sequence_depth_rows.append({'dataset': dataset, 'sequence': sequence, 'depth': depth,
                                        'frames': len(sample), 'mean_iou': float(np.mean(sample)),
                                        'median_iou': float(np.median(sample))})
    save_csv(args.output_dir / 'depth_summary.csv', depth_rows)
    save_csv(args.output_dir / 'sequence_depth_summary.csv', sequence_depth_rows)

    uplift_rows = []
    for d1, d2 in pairs:
        values = uplift[d1, d2]
        positive, negative = values[values > 0], values[values < 0]
        s = summarize(values)
        row = {'dataset': dataset, 'from_depth': d1, 'to_depth': d2,
               'pair_type': 'adjacent' if d2 == d1 + 1 else 'fixed', 'frames': len(values),
               **{f'{name}_delta': value for name, value in s.items()}}
        for threshold in (0, 0.01, 0.02, 0.05, 0.10, 0.20):
            row[f'p_delta_gt_{threshold:.2f}'] = fraction(values, lambda a, t=threshold: a > t)
        for threshold in (0.01, 0.02):
            row[f'p_abs_delta_lt_{threshold:.2f}'] = fraction(values, lambda a, t=threshold: abs(a) < t)
        for threshold in (0, -0.01, -0.05, -0.10):
            row[f'p_delta_lt_{threshold:.2f}'] = fraction(values, lambda a, t=threshold: a < t)
        row['mean_delta_given_positive'] = float(np.mean(positive)) if len(positive) else float('nan')
        row['mean_delta_given_negative'] = float(np.mean(negative)) if len(negative) else float('nan')
        uplift_rows.append(row)
    save_csv(args.output_dir / 'uplift_summary.csv', uplift_rows)

    difficulty_rows = []
    difficulty_correlation = []
    confidence_correlation = []
    for d1, d2 in pairs:
        current = y[:, index[d1]]
        delta = uplift[d1, d2]
        hard, easy = current < 0.5, current >= 0.5
        high, low = delta > 0.05, delta <= 0.05
        n_hard, n_easy = int(hard.sum()), int(easy.sum())
        difficulty_rows.append({'dataset': dataset, 'from_depth': d1, 'to_depth': d2,
                                'frames': len(current), 'n_hard': n_hard, 'n_easy': n_easy,
                                'p_hard': float(hard.mean()), 'p_easy': float(easy.mean()),
                                'p_high_uplift_given_hard': float(high[hard].mean()) if n_hard else float('nan'),
                                'p_low_uplift_given_hard': float(low[hard].mean()) if n_hard else float('nan'),
                                'p_high_uplift_given_easy': float(high[easy].mean()) if n_easy else float('nan'),
                                'p_low_uplift_given_easy': float(low[easy].mean()) if n_easy else float('nan'),
                                'p_hard_and_high_uplift': float(np.mean(hard & high)),
                                'p_hard_and_low_uplift': float(np.mean(hard & low)),
                                'p_easy_and_high_uplift': float(np.mean(easy & high)),
                                'p_easy_and_low_uplift': float(np.mean(easy & low)),
                                'difficulty_mismatch_rate': float(np.mean((hard & low) | (easy & high)))})
        difficulty_correlation.append({'dataset': dataset, 'from_depth': d1, 'to_depth': d2,
                                       **correlation(current, delta)})
        for field in CONFIDENCES:
            confidence_correlation.append({'dataset': dataset, 'from_depth': d1, 'to_depth': d2,
                                           'confidence_field': field, **correlation(conf[field][:, index[d1]], delta),
                                           'confidence_note': CONFIDENCE_NOTE})
    save_csv(args.output_dir / 'difficulty_uplift_conditional.csv', difficulty_rows)
    save_csv(args.output_dir / 'difficulty_uplift_correlation.csv', difficulty_correlation)
    save_csv(args.output_dir / 'confidence_uplift_correlation.csv', confidence_correlation)

    oracle = {epsilon: oracle_depths(y, depths, full_index, epsilon) for epsilon in EPSILONS}
    oracle_frame_rows = []
    for i, key in enumerate(keys):
        oracle_frame_rows.append({'dataset': dataset, 'sequence': key[1], 'frame': key[2],
                                  'full_iou': full[i], **{f'oracle_depth_epsilon_{e:.2f}': int(oracle[e][i])
                                                     for e in EPSILONS}})
    save_csv(args.output_dir / 'oracle_budget.csv', oracle_frame_rows)
    stratified_rows = []
    for label, predicate in STRATA:
        mask = predicate(full)
        for epsilon in EPSILONS:
            stratified_rows.append(oracle_summary(dataset, label, epsilon, oracle[epsilon][mask]))
    save_csv(args.output_dir / 'oracle_summary_stratified.csv', stratified_rows)

    oracle_group_rows = []
    uplift_group_rows = []
    for label, predicate in GROUPS:
        mask = predicate(full)
        for epsilon in EPSILONS:
            row = oracle_summary(dataset, label, epsilon, oracle[epsilon][mask])
            row['p_depth_1_to_6'] = row['p_depth_le_6']
            row['p_depth_7_to_8'] = fraction(oracle[epsilon][mask], lambda a: (a >= 7) & (a <= 8))
            row['p_depth_9_to_10'] = fraction(oracle[epsilon][mask], lambda a: (a >= 9) & (a <= 10))
            row['p_depth_11_to_12'] = fraction(oracle[epsilon][mask], lambda a: (a >= 11) & (a <= 12))
            oracle_group_rows.append(row)
        for d1, d2 in FIXED_PAIRS:
            values = uplift[d1, d2][mask]
            uplift_group_rows.append({'dataset': dataset, 'group': label, 'from_depth': d1, 'to_depth': d2,
                                      'frames': len(values), 'mean_delta': float(np.mean(values)) if len(values) else float('nan'),
                                      'median_delta': float(np.median(values)) if len(values) else float('nan'),
                                      'p_positive': fraction(values, lambda a: a > 0),
                                      'p_negative': fraction(values, lambda a: a < 0)})
    save_csv(args.output_dir / 'oracle_by_full_iou_group.csv', oracle_group_rows)
    save_csv(args.output_dir / 'uplift_by_full_iou_group.csv', uplift_group_rows)

    utility_rows = []
    for cost_lambda in LAMBDAS:
        utilities = y - cost_lambda * depths[None, :]
        chosen_index = np.argmax(utilities, axis=1)  # ascending depths ensure shallow tie-break
        chosen_depth = depths[chosen_index]
        chosen_iou = y[np.arange(len(y)), chosen_index]
        utility_rows.append({'dataset': dataset, 'lambda': cost_lambda, 'frames': len(y),
                             'mean_selected_depth': float(np.mean(chosen_depth)),
                             'median_selected_depth': float(np.median(chosen_depth)),
                             'mean_selected_iou': float(np.mean(chosen_iou)),
                             'mean_full_depth_iou': float(np.mean(full)),
                             'mean_iou_loss_vs_depth12': float(np.mean(full - chosen_iou)),
                             **{f'p_selected_depth_le_{t}': fraction(chosen_depth, lambda a, t=t: a <= t)
                                for t in (6, 8, 9, 10)},
                             'p_selected_depth_eq_12': fraction(chosen_depth, lambda a: a == 12)})
    save_csv(args.output_dir / 'oracle_utility_curve.csv', utility_rows)

    step_delta = np.diff(y, axis=1)
    best_index = np.argmax(y, axis=1)
    max_positive_index = np.argmax(step_delta, axis=1)
    max_positive = np.maximum(np.max(step_delta, axis=1), 0)
    max_negative = np.minimum(np.min(step_delta, axis=1), 0)
    frame_rows = []
    early, late, hopeless, nonmonotonic = [], [], [], []
    for i, key in enumerate(keys):
        row = {'dataset': dataset, 'sequence': key[1], 'frame': key[2],
               **{f'iou_{depth}': y[i, index[depth]] for depth in depths_list},
               'best_depth': int(depths[best_index[i]]), 'best_iou': float(y[i, best_index[i]]),
               'full_iou': float(full[i]),
               **{f'earliest_depth_within_{e:.2f}_of_full': int(oracle[e][i]) for e in EPSILONS},
               'number_of_negative_steps': int(np.sum(step_delta[i] < 0)),
               'maximum_positive_step': float(max_positive[i]),
               'maximum_negative_step': float(max_negative[i]),
               'depth_of_maximum_positive_step': int(depths[max_positive_index[i]]) if max_positive[i] > 0 else '',
               'depth_after_maximum_positive_step': int(depths[max_positive_index[i] + 1]) if max_positive[i] > 0 else ''}
        frame_rows.append(row)
        if full[i] >= 0.5 and oracle[0.02][i] <= 6: early.append(i)
        if full[i] >= 0.5 and max_positive[i] > 0.1 and depths[max_positive_index[i]] >= 8: late.append(i)
        if full[i] < 0.3 and np.max(y[i]) < 0.3 and max_positive[i] < 0.05: hopeless.append(i)
        if max_negative[i] < -0.05: nonmonotonic.append(i)
    save_csv(args.output_dir / 'frame_depth_response.csv', frame_rows)
    patterns = [('early_saturation', early), ('late_gain', late), ('hopeless', hopeless),
                ('non_monotonic', nonmonotonic)]
    pattern_rows = [{'dataset': dataset, 'pattern': name, 'frames': len(indices),
                     'fraction_of_frames': len(indices) / len(y), 'definition': definition}
                    for (name, indices), definition in zip(patterns, (
                        'full_iou>=0.5 and earliest depth within 0.02 of full <=6',
                        'full_iou>=0.5 and largest adjacent uplift >0.1 starts at depth>=8',
                        'full_iou<0.3 and max IoU across depths<0.3 and max adjacent uplift<0.05',
                        'at least one adjacent uplift <-0.05'))]
    save_csv(args.output_dir / 'depth_response_pattern_summary.csv', pattern_rows)

    sequence_rows = []
    for sequence in sorted(set(sequences)):
        mask = sequences == sequence
        chosen = oracle[0.02][mask]
        success = mask & (full >= 0.5)
        row = {'dataset': dataset, 'sequence': sequence, 'frames': int(mask.sum()),
               'mean_iou_12': float(np.mean(full[mask])), 'median_iou_12': float(np.median(full[mask])),
               'oracle_mean_depth_epsilon_0.02': float(np.mean(chosen)),
               'oracle_median_depth_epsilon_0.02': float(np.median(chosen)),
               'full_success_frames': int(success.sum()),
               'oracle_mean_depth_full_success_epsilon_0.02':
                   float(np.mean(oracle[0.02][success])) if success.any() else float('nan'),
               **{f'p_oracle_depth_le_{t}': fraction(chosen, lambda a, t=t: a <= t)
                  for t in (6, 8, 9, 10)}}
        for d1, d2 in FIXED_PAIRS:
            tag = f'{d1}_to_{d2}'
            current = y[mask, index[d1]]
            delta = uplift[d1, d2][mask]
            hard = current < 0.5
            high = delta > 0.05
            row[f'mean_uplift_{tag}'] = float(np.mean(delta))
            row[f'p_high_uplift_{tag}'] = float(np.mean(high))
            row[f'difficulty_mismatch_rate_{tag}'] = float(np.mean((hard & ~high) | (~hard & high)))
        sequence_rows.append(row)
    save_csv(args.output_dir / 'sequence_uplift_summary.csv', sequence_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(depths, [row['mean_iou'] for row in depth_rows], marker='o', label='Mean IoU')
    ax.plot(depths, [row['median_iou'] for row in depth_rows], marker='o', label='Median IoU')
    ax.fill_between(depths, [row['q25_iou'] for row in depth_rows],
                    [row['q75_iou'] for row in depth_rows], alpha=0.2, label='25%-75%')
    ax.set(xlabel='Encoder depth', ylabel='IoU', xticks=depths); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / 'depth_vs_iou.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(5, 4, figsize=(14, 14))
    for ax, pair in zip(axes.flat, pairs):
        ax.hist(uplift[pair], bins=40, color='#4477aa')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(f'{pair[0]} to {pair[1]}'); ax.set_xlabel('IoU delta')
    for ax in list(axes.flat)[len(pairs):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(args.output_dir / 'uplift_distribution.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    labels = [f'{a}->{b}' for a, b in pairs]
    x = np.arange(len(pairs))
    for field, label in [('p_delta_gt_0.05', 'Delta > 0.05'),
                         ('p_delta_lt_0.00', 'Delta < 0'),
                         ('p_delta_lt_-0.05', 'Delta < -0.05'),
                         ('p_abs_delta_lt_0.02', '|Delta| < 0.02')]:
        ax.plot(x, [row[field] for row in uplift_rows], marker='o', label=label)
    ax.set_xticks(x, labels, rotation=55, ha='right'); ax.set_ylim(0, 1)
    ax.set_ylabel('Frame proportion'); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / 'uplift_positive_negative_ratio.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, pair in zip(axes.flat, SCATTER_PAIRS):
        current, delta = y[:, index[pair[0]]], uplift[pair]
        ax.scatter(current, delta, s=4, alpha=0.12)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
        ax.axhline(0.05, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set(title=f'{pair[0]} to {pair[1]}', xlabel=f'IoU at depth {pair[0]}', ylabel='Uplift')
    fig.tight_layout(); fig.savefig(args.output_dir / 'difficulty_vs_uplift_quadrants.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in utility_rows:
        ax.scatter(row['mean_selected_depth'], row['mean_selected_iou'], s=50,
                   label=f"lambda={row['lambda']:g}")
    ax.plot([row['mean_selected_depth'] for row in utility_rows],
            [row['mean_selected_iou'] for row in utility_rows], alpha=0.5)
    ax.set(xlabel='Mean oracle selected depth', ylabel='Mean selected IoU')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / 'oracle_accuracy_compute_tradeoff.png', dpi=160); plt.close(fig)

    rng = np.random.default_rng(args.random_seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, candidates) in zip(axes.flat, patterns):
        selection = rng.choice(candidates, size=min(20, len(candidates)), replace=False) if candidates else []
        for i in selection:
            ax.plot(depths, y[i], alpha=0.5, linewidth=1)
        ax.set(title=f'{name}: {len(candidates)} frames, {len(selection)} shown', xlabel='Encoder depth',
               ylabel='IoU', ylim=(-0.05, 1), xticks=depths)
    fig.tight_layout(); fig.savefig(args.output_dir / 'depth_response_examples.png', dpi=160); plt.close(fig)

    with (args.output_dir / 'analysis_metadata.json').open('w') as file:
        json.dump({'dataset': dataset, 'frames_analyzed': len(y), 'pairs': pairs,
                   'confidence_note': CONFIDENCE_NOTE,
                   'oracle_note': 'Open-loop GT oracle upper bound; cost(d)=d; no online routing',
                   'aggregation_note': 'Dataset-level summaries weight frames equally; sequence_uplift_summary.csv keeps per-sequence results.',
                   'statistics': 'Population standard deviation; empirical linear-interpolation quantiles; unadjusted two-sided correlation p-values. P-values treat frames as independent despite temporal dependence.'},
                  file, indent=2)
    try:
        from tracking.encoder_uplift_extended import run_extended_analysis
    except ModuleNotFoundError:
        from encoder_uplift_extended import run_extended_analysis
    raw = {field: np.array([[frames[key][depth]['raw'][field] for depth in depths] for key in keys])
           for field in RAW_SUMMARY_FIELDS}
    similarity = {field: np.array([[frames[key][depth]['similarity'][field] for depth in depths]
                                   for key in keys]) for field in SIMILARITY_FIELDS}
    gt = np.array([frames[key][args.full_depth]['gt'] for key in keys])
    pred = np.array([[frames[key][depth]['box'] for depth in depths] for key in keys])
    extended = run_extended_analysis(args.output_dir, dataset, depths, keys, y, conf, raw,
                                     similarity, gt, pred, oracle, depth_rows, args.random_seed)
    with (args.output_dir / 'extended_metadata.json').open('w') as file:
        json.dump(extended, file, indent=2)
    print(f'Data quality passed: {quality["sequences"]} sequences, {quality["frames"]} frames, {quality["rows"]} rows')
    print(f'Max box error versus saved sweep/baseline: {quality["max_abs_box_error"]}')
    print(f'Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
