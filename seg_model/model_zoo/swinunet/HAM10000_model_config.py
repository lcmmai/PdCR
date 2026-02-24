from seg_model.total_config import HAM10000_general_config


class setting_config(HAM10000_general_config):
    """
    the config of training setting.
    """

    network = 'swinunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'img_size': 256,
        'patch_size': 4,
        'in_chans': 3,
        'embed_dim': 96,
        'depths': [2, 2, 2, 2],
        'decoder_depths': [2, 2, 2, 1],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'drop_path_rate': 0.2,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
    }

    work_dir = 'seg_model/model_zoo/swinunet/'
