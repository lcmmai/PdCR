from torch.utils.data import DataLoader
from .dataset_ham10000 import dataset_ham10000
from .dataset_fives import dataset_fives


def get_loader(config):
    if config.data_name == "HAM10000":
        train_dataset = dataset_ham10000(config.data_path, config.image_size, mode="train")
        test_loader = dataset_ham10000(config.data_path, config.image_size, mode="test")
    elif config.data_name == "FIVES":
        train_dataset = dataset_fives(config.data_path, config.image_size, mode="train")
        test_loader = dataset_fives(config.data_path, config.image_size, mode="test")
    else:
        raise ValueError("Dataset not supported yet.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        test_loader,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers
    )

    return train_loader, test_loader




