from seg_model.total_config import FIVES_general_config


class setting_config(FIVES_general_config):
    """
    the config of training setting.
    """

    network = 'hrnet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
    }

    work_dir = 'seg_model/model_zoo/hrnet/'
