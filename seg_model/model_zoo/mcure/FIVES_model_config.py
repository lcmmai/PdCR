from seg_model.total_config import FIVES_general_config


class setting_config(FIVES_general_config):
    """
    the config of training setting.
    """

    network = 'mcure'
    model_config = {
        'num_classes': 1,
    }

    work_dir = 'seg_model/model_zoo/mcure/'
