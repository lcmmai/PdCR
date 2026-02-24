from seg_model.total_config import HAM10000_general_config


class setting_config(HAM10000_general_config):
    """
    the config of training setting.
    """

    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
    }

    work_dir = 'seg_model/model_zoo/vmunet/'
