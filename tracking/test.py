import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, sequences=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequences is not None:
        dataset = [dataset[name] for name in sequences]
    elif sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (otb, nfs, uav, got10k_test, '
                                                                          'lasot, trackingnet, lasot_extension_subset, tnl2k,'
                                                                          'lasot_lang, otb99_lang).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--sequences', type=str, default=None,
                        help='Comma-separated sequence names for a small diagnostic run.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--encoder_depth', type=int, default=None)
    parser.add_argument('--encoder_sweep', action='store_true')
    parser.add_argument('--sweep_depths', type=str, default=None, help='Comma-separated encoder depths.')
    parser.add_argument('--save_sweep_results', action='store_true')
    parser.add_argument('--measure_encoder_latency', action='store_true')

    args = parser.parse_args()
    if args.sequence is not None and args.sequences is not None:
        parser.error('--sequence and --sequences cannot be used together')
    sequence_names = [name.strip() for name in args.sequences.split(',')] if args.sequences else None
    if sequence_names is not None and (not all(sequence_names) or len(set(sequence_names)) != len(sequence_names)):
        parser.error('--sequences must contain unique, nonempty names')
    if args.encoder_depth is not None:
        os.environ['SEQTRACK_ENCODER_DEPTH'] = str(args.encoder_depth)
    if args.encoder_sweep:
        os.environ['SEQTRACK_ENCODER_SWEEP'] = '1'
    if args.sweep_depths:
        os.environ['SEQTRACK_SWEEP_DEPTHS'] = args.sweep_depths
    if args.save_sweep_results:
        os.environ['SEQTRACK_SAVE_SWEEP'] = '1'
    if args.measure_encoder_latency:
        os.environ['SEQTRACK_MEASURE_LATENCY'] = '1'

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, sequences=sequence_names)


if __name__ == '__main__':
    main()
