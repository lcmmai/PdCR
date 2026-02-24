from seg_model.total_config import HAM10000_general_config


class setting_config(HAM10000_general_config):
    """
    the config of training setting.
    """

    network = 'unetpp'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
    }

    work_dir = 'seg_model/model_zoo/unetpp/'
