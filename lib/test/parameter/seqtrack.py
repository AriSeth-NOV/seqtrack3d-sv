from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.seqtrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/seqtrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join("./checkpoints/train/seqtrack/%s/SEQTRACK_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))
    if not os.path.isfile(params.checkpoint):
        trained_checkpoint = os.path.join(save_dir, 'checkpoints/train/seqtrack', yaml_name,
                                          'SEQTRACK_ep%04d.pth.tar' % cfg.TEST.EPOCH)
        if os.path.isfile(trained_checkpoint):
            params.checkpoint = trained_checkpoint

    # whether to save boxes from all queries
    params.save_all_boxes = False

    params.encoder_depth = 12
    params.encoder_sweep = False
    params.sweep_depths = list(range(1, 13))
    params.save_sweep_results = False
    params.measure_encoder_latency = False
    # Optional CLI overrides, inherited by evaluation worker processes.
    params.encoder_depth_explicit = 'SEQTRACK_ENCODER_DEPTH' in os.environ
    params.encoder_depth = int(os.environ.get('SEQTRACK_ENCODER_DEPTH', params.encoder_depth))
    params.encoder_sweep = os.environ.get('SEQTRACK_ENCODER_SWEEP', '0') == '1'
    if os.environ.get('SEQTRACK_SWEEP_DEPTHS'):
        params.sweep_depths = [int(d) for d in os.environ['SEQTRACK_SWEEP_DEPTHS'].split(',')]
    params.save_sweep_results = os.environ.get('SEQTRACK_SAVE_SWEEP', '0') == '1'
    params.measure_encoder_latency = os.environ.get('SEQTRACK_MEASURE_LATENCY', '0') == '1'

    return params
