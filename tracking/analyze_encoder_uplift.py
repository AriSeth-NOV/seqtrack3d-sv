"""Analyze complete-sequence counterfactual encoder sweeps for SeqTrack."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def average(values):
    return mean(values) if values else float('nan')


def middle(values):
    return median(values) if values else float('nan')


def ratio(values, predicate):
    return sum(predicate(value) for value in values) / len(values) if values else float('nan')


def save_csv(path, columns, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_sweeps(paths, full_depth):
    frames = defaultdict(dict)
    row_count = 0
    duplicates = []
    invalid = []
    csv_files = []
    for path in paths:
        csv_files.extend(sorted(path.rglob('encoder_sweep.csv')) if path.is_dir() else [path])
    csv_files = sorted(set(csv_files))
    if not csv_files:
        raise ValueError('No encoder_sweep.csv files found')
    for path in csv_files:
        with path.open(newline='') as file:
            for row in csv.DictReader(file):
                row_count += 1
                try:
                    sequence = row['sequence']
                    frame = int(row['frame'])
                    depth = int(row['depth'])
                    iou = float(row['iou'])
                    box = tuple(float(row[f'pred_{axis}']) for axis in 'xywh')
                    if not sequence or frame < 1 or depth < 1 or not math.isfinite(iou) or not all(map(math.isfinite, box)):
                        raise ValueError('invalid sequence, frame, depth, IoU or box')
                except (KeyError, TypeError, ValueError) as error:
                    invalid.append({'file': str(path), 'row': row_count, 'reason': str(error)})
                    continue
                key = (sequence, frame)
                if depth in frames[key]:
                    duplicates.append({'sequence': sequence, 'frame': frame, 'depth': depth,
                                       'file': str(path)})
                    continue
                frames[key][depth] = {'iou': iou, 'box': box, 'confidence': row.get('conf_mean', '')}
    all_depths = sorted({depth for frame in frames.values() for depth in frame})
    if full_depth not in all_depths:
        raise ValueError(f'No depth {full_depth} rows found')
    incomplete = [{'sequence': sequence, 'frame': frame, 'depths': sorted(data)}
                  for (sequence, frame), data in frames.items() if sorted(data) != all_depths]
    quality = {'csv_files': len(csv_files), 'rows': row_count, 'sequences': len({key[0] for key in frames}),
               'frames': len(frames), 'depths': all_depths, 'invalid_rows': len(invalid),
               'duplicate_rows': len(duplicates), 'incomplete_frames': len(incomplete),
               'invalid_examples': invalid[:20], 'duplicate_examples': duplicates[:20],
               'incomplete_examples': incomplete[:20]}
    return frames, all_depths, quality


def check_sequence_coverage(frames, dataset_root, expected_sequences):
    observed = {key[0] for key in frames}
    missing_sequences = sorted(set(expected_sequences) - observed) if expected_sequences else []
    unexpected_sequences = sorted(observed - set(expected_sequences)) if expected_sequences else []
    details = []
    if dataset_root:
        for sequence in sorted(observed):
            annotation = dataset_root / sequence / 'groundTruth.rect'
            if not annotation.is_file():
                details.append({'sequence': sequence, 'status': 'missing_annotation'})
                continue
            with annotation.open() as file:
                expected_count = sum(bool(line.strip()) for line in file)
            actual = {frame for name, frame in frames if name == sequence}
            missing = sorted(set(range(1, expected_count)) - actual)
            extra = sorted(actual - set(range(1, expected_count)))
            details.append({'sequence': sequence, 'expected_tracked_frames': expected_count - 1,
                            'observed_frames': len(actual), 'missing_frames': len(missing),
                            'extra_frames': len(extra), 'missing_examples': missing[:20],
                            'extra_examples': extra[:20],
                            'status': 'complete' if not missing and not extra else 'incomplete'})
    return {'missing_sequences': missing_sequences, 'unexpected_sequences': unexpected_sequences,
            'sequence_coverage': details}


def compare_baseline(frames, baseline_dir, full_depth, tolerance):
    comparisons = []
    for sequence in sorted({key[0] for key in frames}):
        path = baseline_dir / f'{sequence}.txt'
        if not path.is_file():
            comparisons.append({'sequence': sequence, 'status': 'missing_baseline',
                                'compared_frames': 0, 'max_abs_box_error': ''})
            continue
        with path.open() as file:
            boxes = [[float(value) for value in line.split()] for line in file if line.strip()]
        seq_frames = [(frame, values) for (name, frame), values in frames.items() if name == sequence]
        errors = []
        missing = 0
        for frame, values in seq_frames:
            if frame >= len(boxes) or full_depth not in values or len(boxes[frame]) != 4:
                missing += 1
                continue
            errors.append(max(abs(a - b) for a, b in zip(boxes[frame], values[full_depth]['box'])))
        maximum = max(errors) if errors else float('nan')
        comparisons.append({'sequence': sequence,
                            'status': 'missing_frames' if missing else ('match' if maximum <= tolerance else 'mismatch'),
                            'compared_frames': len(errors), 'max_abs_box_error': maximum})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Sweep CSV files or a result directory')
    parser.add_argument('--output_dir', type=Path, default=Path('encoder_uplift_analysis'))
    parser.add_argument('--full_depth', type=int, default=12)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.02, 0.05])
    parser.add_argument('--cost_lambda', type=float, default=0.0,
                        help='Utility oracle uses IoU - lambda * depth; depth is a cost proxy')
    parser.add_argument('--baseline_results_dir', type=Path, default=None,
                        help='Optional baseline bbox directory for frame-by-frame depth-12 comparison')
    parser.add_argument('--sweep_bbox_results_dir', type=Path, default=None,
                        help='Sweep bbox directory for verifying saved trajectory equals depth-12 CSV boxes')
    parser.add_argument('--dataset_root', type=Path, default=None,
                        help='SV248S root for checking that every sequence frame was swept')
    parser.add_argument('--expected_sequences', type=str, default=None,
                        help='Comma-separated names expected in this result directory')
    parser.add_argument('--box_tolerance', type=float, default=1e-4)
    parser.add_argument('--difficulty_threshold', type=float, default=0.5)
    parser.add_argument('--uplift_threshold', type=float, default=0.05)
    args = parser.parse_args()
    if not args.epsilons or any(e < 0 for e in args.epsilons):
        parser.error('epsilons must be nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, depths, quality = load_sweeps(args.inputs, args.full_depth)
    if not frames:
        parser.error('No valid sweep rows found')
    expected_sequences = args.expected_sequences.split(',') if args.expected_sequences else None
    quality.update(check_sequence_coverage(frames, args.dataset_root, expected_sequences))
    complete = {key: values for key, values in frames.items() if sorted(values) == depths}
    quality['complete_frames'] = len(complete)
    quality['analysis_uses_complete_frames_only'] = True
    if args.baseline_results_dir:
        comparisons = compare_baseline(frames, args.baseline_results_dir, args.full_depth, args.box_tolerance)
        save_csv(args.output_dir / 'baseline_comparison.csv',
                 ['sequence', 'status', 'compared_frames', 'max_abs_box_error'], comparisons)
        quality['baseline_status_counts'] = dict(Counter(row['status'] for row in comparisons))
    if args.sweep_bbox_results_dir:
        comparisons = compare_baseline(frames, args.sweep_bbox_results_dir, args.full_depth, args.box_tolerance)
        save_csv(args.output_dir / 'sweep_trajectory_comparison.csv',
                 ['sequence', 'status', 'compared_frames', 'max_abs_box_error'], comparisons)
        quality['sweep_trajectory_status_counts'] = dict(Counter(row['status'] for row in comparisons))
    with (args.output_dir / 'data_quality.json').open('w') as file:
        json.dump(quality, file, indent=2)
    if not complete:
        parser.error('No frames contain all measured depths; see data_quality.json')

    depth_rows = []
    sequence_depth_rows = []
    for depth in depths:
        ious = [frame[depth]['iou'] for frame in complete.values()]
        depth_rows.append({'depth': depth, 'frames': len(ious), 'mean_iou': average(ious),
                           'median_iou': middle(ious)})
        for sequence in sorted({key[0] for key in complete}):
            seq_ious = [frame[depth]['iou'] for (name, _), frame in complete.items() if name == sequence]
            sequence_depth_rows.append({'sequence': sequence, 'depth': depth, 'frames': len(seq_ious),
                                        'mean_iou': average(seq_ious), 'median_iou': middle(seq_ious)})
    save_csv(args.output_dir / 'depth_summary.csv', ['depth', 'frames', 'mean_iou', 'median_iou'], depth_rows)
    save_csv(args.output_dir / 'sequence_depth_summary.csv',
             ['sequence', 'depth', 'frames', 'mean_iou', 'median_iou'], sequence_depth_rows)

    steps = list(zip(depths[:-1], depths[1:]))
    deltas_by_step = {(d1, d2): [frame[d2]['iou'] - frame[d1]['iou'] for frame in complete.values()]
                      for d1, d2 in steps}
    uplift_rows = []
    for d1, d2 in steps:
        values = deltas_by_step[(d1, d2)]
        negatives = [value for value in values if value < 0]
        uplift_rows.append({'from_depth': d1, 'to_depth': d2, 'frames': len(values),
                            'mean_delta': average(values), 'median_delta': middle(values),
                            'p_delta_gt_0.02': ratio(values, lambda x: x > 0.02),
                            'p_delta_gt_0.05': ratio(values, lambda x: x > 0.05),
                            'p_delta_gt_0.10': ratio(values, lambda x: x > 0.10),
                            'p_delta_gt_0.20': ratio(values, lambda x: x > 0.20),
                            'p_abs_delta_lt_0.01': ratio(values, lambda x: abs(x) < 0.01),
                            'p_delta_lt_0': ratio(values, lambda x: x < 0),
                            'mean_delta_given_negative': average(negatives)})
    uplift_columns = ['from_depth', 'to_depth', 'frames', 'mean_delta', 'median_delta',
                      'p_delta_gt_0.02', 'p_delta_gt_0.05', 'p_delta_gt_0.10',
                      'p_delta_gt_0.20', 'p_abs_delta_lt_0.01', 'p_delta_lt_0',
                      'mean_delta_given_negative']
    save_csv(args.output_dir / 'uplift_summary.csv', uplift_columns, uplift_rows)

    oracle_rows = []
    for (sequence, frame_number), values in sorted(complete.items()):
        full_iou = values[args.full_depth]['iou']
        row = {'sequence': sequence, 'frame': frame_number, 'full_iou': full_iou}
        for epsilon in args.epsilons:
            row[f'epsilon_{epsilon:g}_depth'] = min(
                depth for depth in depths if values[depth]['iou'] >= full_iou - epsilon)
        utility_depth = min(depths, key=lambda depth: (-(values[depth]['iou'] - args.cost_lambda * depth), depth))
        row['utility_depth'] = utility_depth
        row['utility_iou'] = values[utility_depth]['iou']
        oracle_rows.append(row)
    oracle_columns = ['sequence', 'frame', 'full_iou'] + [f'epsilon_{e:g}_depth' for e in args.epsilons] + [
        'utility_depth', 'utility_iou']
    save_csv(args.output_dir / 'oracle_budget.csv', oracle_columns, oracle_rows)
    oracle_summary = []
    for epsilon in args.epsilons:
        key = f'epsilon_{epsilon:g}_depth'
        chosen = [row[key] for row in oracle_rows]
        oracle_summary.append({'epsilon': epsilon, 'frames': len(chosen), 'mean_depth': average(chosen),
                               'median_depth': middle(chosen),
                               'p_depth_le_6': ratio(chosen, lambda d: d <= 6),
                               'p_depth_le_8': ratio(chosen, lambda d: d <= 8),
                               'p_depth_le_10': ratio(chosen, lambda d: d <= 10),
                               'p_depth_11_to_12': ratio(chosen, lambda d: 11 <= d <= 12)})
    save_csv(args.output_dir / 'oracle_summary.csv',
             ['epsilon', 'frames', 'mean_depth', 'median_depth', 'p_depth_le_6', 'p_depth_le_8',
              'p_depth_le_10', 'p_depth_11_to_12'], oracle_summary)

    pairs = [(d1, d2) for d1, d2 in [(6, 8), (8, 10), (10, 12), (6, 12)] if d1 in depths and d2 in depths]
    difficulty_rows = []
    for d1, d2 in pairs:
        samples = [(values[d1]['iou'], values[d2]['iou'] - values[d1]['iou']) for values in complete.values()]
        hard = args.difficulty_threshold
        useful = args.uplift_threshold
        difficulty_rows.append({'from_depth': d1, 'to_depth': d2, 'frames': len(samples),
                                'p_hard_high_uplift': ratio(samples, lambda p: p[0] < hard and p[1] > useful),
                                'p_hard_low_uplift': ratio(samples, lambda p: p[0] < hard and p[1] <= useful),
                                'p_easy_high_uplift': ratio(samples, lambda p: p[0] >= hard and p[1] > useful),
                                'p_easy_low_uplift': ratio(samples, lambda p: p[0] >= hard and p[1] <= useful)})
    save_csv(args.output_dir / 'difficulty_uplift_summary.csv',
             ['from_depth', 'to_depth', 'frames', 'p_hard_high_uplift', 'p_hard_low_uplift',
              'p_easy_high_uplift', 'p_easy_low_uplift'], difficulty_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(depths, [row['mean_iou'] for row in depth_rows], marker='o', label='Mean IoU')
    plt.plot(depths, [row['median_iou'] for row in depth_rows], marker='o', label='Median IoU')
    plt.xlabel('Encoder depth'); plt.ylabel('IoU'); plt.xticks(depths); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / 'depth_vs_iou.png', dpi=160); plt.close()

    fig, axes = plt.subplots(math.ceil(len(steps) / 3), 3, figsize=(12, 3 * math.ceil(len(steps) / 3)))
    for ax, (d1, d2) in zip(axes.flat, steps):
        ax.hist(deltas_by_step[(d1, d2)], bins=40, color='#4477aa')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(f'{d1} to {d2}'); ax.set_xlabel('IoU delta'); ax.set_ylabel('Frames')
    for ax in list(axes.flat)[len(steps):]:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(args.output_dir / 'uplift_distribution.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for threshold in (0.02, 0.05, 0.10, 0.20):
        ax.plot([d1 for d1, _ in steps], [row[f'p_delta_gt_{threshold:.2f}'] for row in uplift_rows],
                marker='o', label=f'Delta > {threshold:.2f}')
    ax.plot([d1 for d1, _ in steps], [row['p_delta_lt_0'] for row in uplift_rows],
            marker='x', linestyle='--', label='Delta < 0')
    ax.plot([d1 for d1, _ in steps], [row['p_abs_delta_lt_0.01'] for row in uplift_rows],
            marker='x', linestyle=':', label='|Delta| < 0.01')
    ax.set_xlabel('From depth'); ax.set_ylabel('Frame proportion'); ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(args.output_dir / 'uplift_positive_ratio.png', dpi=160); plt.close(fig)

    if pairs:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, (d1, d2) in zip(axes.flat, pairs):
            x = [values[d1]['iou'] for values in complete.values()]
            y = [values[d2]['iou'] - values[d1]['iou'] for values in complete.values()]
            ax.scatter(x, y, s=5, alpha=0.18)
            ax.axhline(args.uplift_threshold, color='red', linestyle='--', linewidth=1)
            ax.axvline(args.difficulty_threshold, color='red', linestyle='--', linewidth=1)
            ax.axhline(0, color='black', linewidth=0.7)
            ax.set_title(f'{d1} to {d2}'); ax.set_xlabel(f'IoU at depth {d1}'); ax.set_ylabel('IoU uplift')
        for ax in list(axes.flat)[len(pairs):]:
            ax.axis('off')
        fig.tight_layout(); fig.savefig(args.output_dir / 'difficulty_vs_uplift.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for epsilon in args.epsilons:
        chosen = [row[f'epsilon_{epsilon:g}_depth'] for row in oracle_rows]
        ax.plot(depths, [ratio(chosen, lambda d, maximum=depth: d <= maximum) for depth in depths],
                marker='o', label=f'Epsilon={epsilon:g}')
    ax.set_xlabel('Maximum allowed depth'); ax.set_ylabel('Frames with oracle depth at most d')
    ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / 'oracle_depth_cdf.png', dpi=160); plt.close(fig)

    print(f"Sequences: {quality['sequences']}; complete frames: {len(complete)}; CSV files: {quality['csv_files']}")
    print(f"Invalid rows: {quality['invalid_rows']}; duplicate rows: {quality['duplicate_rows']}; incomplete frames: {quality['incomplete_frames']}")
    print(f"Missing sequences: {quality['missing_sequences']}; incomplete sequences: "
          f"{[row['sequence'] for row in quality['sequence_coverage'] if row['status'] != 'complete']}")
    if args.baseline_results_dir:
        print(f"Baseline comparison: {quality['baseline_status_counts']}")
    if args.sweep_bbox_results_dir:
        print(f"Sweep trajectory comparison: {quality['sweep_trajectory_status_counts']}")
    for row in oracle_summary:
        print(f"Oracle epsilon={row['epsilon']:g}: mean depth={row['mean_depth']:.2f}, "
              f"P(depth<=6)={row['p_depth_le_6']:.3f}, P(depth<=8)={row['p_depth_le_8']:.3f}, "
              f"P(depth<=10)={row['p_depth_le_10']:.3f}")
    print(f'Output: {args.output_dir}')


if __name__ == '__main__':
    main()
