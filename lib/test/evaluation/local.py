from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/douzhilei/datasets/got10k_lmdb'
    settings.got10k_path = '/home/douzhilei/datasets/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/douzhilei/datasets/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/douzhilei/datasets/lasot_lmdb'
    settings.lasot_path = '/home/douzhilei/datasets/lasot'
    settings.lasotlang_path = '/home/douzhilei/datasets/lasot'
    settings.network_path = '/home/douzhilei/seqtrack3d-sv/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/douzhilei/datasets/nfs'
    settings.otb_path = '/home/douzhilei/datasets/OTB2015'
    settings.otblang_path = '/home/douzhilei/datasets/otb_lang'
    settings.prj_dir = '/home/douzhilei/seqtrack3d-sv'
    settings.result_plot_path = '/home/douzhilei/seqtrack3d-sv/output/test/result_plots'
    settings.results_path = '/home/douzhilei/seqtrack3d-sv/output_own_test/tracking_results'
    settings.satsot_path = '/home/douzhilei/datasets/satsot'
    settings.save_dir = '/home/douzhilei/seqtrack3d-sv/output_own_train'
    settings.segmentation_path = '/home/douzhilei/seqtrack3d-sv/output/test/segmentation_results'
    settings.sv248s_test_path = '/home/douzhilei/datasets/sv248/test_sv'
    settings.tc128_path = '/home/douzhilei/datasets/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/douzhilei/datasets/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/douzhilei/datasets/trackingnet'
    settings.uav_path = '/home/douzhilei/datasets/UAV123'
    settings.viso_path = '/home/douzhilei/datasets/viso'
    settings.vot_path = '/home/douzhilei/datasets/VOT2019'
    settings.youtubevos_dir = ''

    return settings

