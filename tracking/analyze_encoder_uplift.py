"""Summarize SeqTrack encoder sweep CSV files without running a tracker."""
import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values) if values else float('nan')


def median(values):
    return statistics.median(values) if values else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='CSV files or result directories')
    parser.add_argument('--output_dir', type=Path, default=Path('encoder_uplift_analysis'))
    parser.add_argument('--epsilon', type=float, default=0.02)
    parser.add_argument('--cost_lambda', type=float, default=0.0)
    args = parser.parse_args()

    paths = []
    for path in args.inputs:
        paths.extend(sorted(path.rglob('encoder_sweep.csv')) if path.is_dir() else [path])
    if not paths:
        parser.error('No encoder_sweep.csv files found')
    rows = []
    for path in paths:
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                for key in ('depth', 'frame'):
                    row[key] = int(row[key])
                for key in ('iou', 'conf_mean', 'delta_to_next_depth'):
                    row[key] = float(row[key]) if row.get(key) else float('nan')
                rows.append(row)

    by_depth = defaultdict(list)
    by_frame = defaultdict(dict)
    for row in rows:
        by_depth[row['depth']].append(row)
        by_frame[(row['sequence'], row['frame'])][row['depth']] = row['iou']

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ['depth', 'mean_iou', 'median_iou', 'mean_confidence', 'median_confidence',
               'mean_delta_to_next', 'median_delta_to_next', 'delta_gt_0.05',
               'delta_gt_0.10', 'delta_gt_0.20']
    summary = []
    for depth, records in sorted(by_depth.items()):
        iou = [r['iou'] for r in records if r['iou'] == r['iou']]
        conf = [r['conf_mean'] for r in records if r['conf_mean'] == r['conf_mean']]
        delta = [r['delta_to_next_depth'] for r in records if r['delta_to_next_depth'] == r['delta_to_next_depth']]
        summary.append(dict(zip(columns, [depth, mean(iou), median(iou), mean(conf), median(conf),
                                          mean(delta), median(delta),
                                          *[sum(v > threshold for v in delta) / len(delta) if delta else float('nan')
                                            for threshold in (0.05, 0.10, 0.20)]])))
    with (args.output_dir / 'depth_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    oracle_rows = []
    for (sequence, frame), ious in sorted(by_frame.items()):
        ious = {d: v for d, v in ious.items() if v == v}
        if not ious:
            continue
        full_depth = max(ious)
        epsilon_depth = min(d for d, v in ious.items() if v >= ious[full_depth] - args.epsilon)
        utility_depth = min(ious, key=lambda d: (-(ious[d] - args.cost_lambda * d), d))
        oracle_rows.append({'sequence': sequence, 'frame': frame, 'full_depth': full_depth,
                            'epsilon_depth': epsilon_depth, 'utility_depth': utility_depth,
                            'full_iou': ious[full_depth], 'epsilon_iou': ious[epsilon_depth],
                            'utility_iou': ious[utility_depth]})
    with (args.output_dir / 'oracle_budget.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'frame', 'full_depth', 'epsilon_depth',
                                               'utility_depth', 'full_iou', 'epsilon_iou', 'utility_iou'])
        writer.writeheader()
        writer.writerows(oracle_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depths = [r['depth'] for r in summary]
    plt.figure()
    plt.plot(depths, [r['mean_iou'] for r in summary], marker='o', label='Mean IoU')
    plt.plot(depths, [r['median_iou'] for r in summary], marker='o', label='Median IoU')
    plt.xlabel('Encoder depth'); plt.ylabel('IoU'); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / 'depth_vs_iou.png'); plt.close()
    valid = [r for r in rows if r['delta_to_next_depth'] == r['delta_to_next_depth']]
    plt.figure()
    plt.hist([r['delta_to_next_depth'] for r in valid], bins=50)
    plt.xlabel('IoU delta to next measured depth'); plt.ylabel('Frames'); plt.tight_layout()
    plt.savefig(args.output_dir / 'uplift_distribution.png'); plt.close()
    plt.figure()
    for threshold in (0.05, 0.10, 0.20):
        plt.plot(depths, [r[f'delta_gt_{threshold:.2f}'] for r in summary], marker='o', label=f'Delta > {threshold:.2f}')
    plt.xlabel('Encoder depth'); plt.ylabel('Frame proportion'); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / 'uplift_positive_ratio.png'); plt.close()
    count = Counter(r['epsilon_depth'] for r in oracle_rows)
    print(f'Frames with valid IoU: {len(oracle_rows)}')
    print(f'Epsilon oracle depth counts: {dict(sorted(count.items()))}')
    print(f'Mean epsilon oracle depth: {mean([r["epsilon_depth"] for r in oracle_rows]):.3f}')
    print(f'Mean utility oracle depth: {mean([r["utility_depth"] for r in oracle_rows]):.3f}')
    print(f'Output: {args.output_dir}')


if __name__ == '__main__':
    main()
