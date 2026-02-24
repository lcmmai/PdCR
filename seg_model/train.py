import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

from pathlib import Path

from seg_model.datasets_.get_dataloader import get_loader
from seg_model.model_utils import get_optimizer, get_scheduler, get_logger, log_config_info, build_model, cal_params_flops
from utils_ import set_seed, test_cal_dice_coefficient


def main(config, device):
    set_seed(0)

    logger = get_logger(config.data_name, config.work_dir)
    log_config_info(config, logger)

    checkpoint_dir = Path(config.work_dir)
    resume_model = Path(config.work_dir) / f'{config.data_name}_latest.pth'

    train_dataloader, test_dataloader = get_loader(config)

    model = build_model(config.network, config.model_config)
    model = model.to(device)

    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    max_dice = 0
    start_epoch = 0

    cal_params_flops(model, config.image_size[0], device, logger)

    if resume_model.exists():
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}'
        # print(log_info)
        logger.info(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs):
        model.train()
        loss_list = []
        for iter, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            images, targets = data["image"].to(device), data["mask"].to(device)

            out = model(images)
            loss = criterion(out, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 调整 max_norm 值

            optimizer.step()

            loss_list.append(loss.item())

            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            if iter % config.print_interval == 0 and iter != 0:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                # print(log_info)
                logger.info(log_info)
        scheduler.step()

        print('#----------Test during training----------#')
        model.eval()
        dice_score = []
        with torch.no_grad():
            for data in test_dataloader:
                img, msk = data["image"].to(device), data["mask"].to(device)

                pred = model(img)

                predicted = (pred[:, :, :, :] > 0.5).float()

                dice = test_cal_dice_coefficient(predicted, msk)
                dice_score.append(dice)

            cated_dice = torch.cat(dice_score, dim=0)
            mean_dice = torch.mean(cated_dice, dim=0).item()

        log_info = f'val epoch: {epoch}, Dice: {mean_dice:.4f}'
        # print(log_info)
        logger.info(log_info)

        if mean_dice > max_dice:
            logger.info(f"Best model changed, is epoch {epoch}")
            # print(f"Best model changed, is epoch {epoch}")
            torch.save(model.state_dict(), checkpoint_dir / f'{config.data_name}_best.pth')
            max_dice = mean_dice

        torch.save(
            {
                'epoch': epoch,
                'max_dice': max_dice,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir / f'{config.data_name}_latest.pth')


if __name__ == "__main__":

    # Run main function with config
    device = "cuda:0"

    from seg_model.model_zoo.unet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.swinunet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.vmunet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.ukan.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.unetpp.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.segformer.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.hrnet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.lightmunet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.segnet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.transunet.HAM10000_model_config import setting_config as config
    # from seg_model.model_zoo.unext.HAM10000_model_config import setting_config as config

    # from seg_model.model_zoo.mcure.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.unet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.swinunet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.vmunet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.ukan.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.unetpp.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.segformer.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.hrnet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.segnet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.lightmunet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.transunet.FIVES_model_config import setting_config as config
    # from seg_model.model_zoo.unext.FIVES_model_config import setting_config as config

    main(config, device)













