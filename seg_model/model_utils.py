import os
import math
import inspect

import torch
import torch.nn as nn

import logging
import logging.handlers
from thop import profile
import torch.nn.functional as F


def cal_params_flops(model, size, device, logger=None):
    input = torch.randn(1, 3, size, size).to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops', flops/1e9)
    print('params', params/1e6)

    total = sum(p.numel() for p in model.parameters())
    # print("Total params: %.2fM" % (total/1e6))
    print(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
    if logger is not None:
        logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
    else:
        print(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class BCE_DiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCE_DiceLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (tensor): The predicted output from the model. Shape should be [b, 1, 256, 256]
            targets (tensor): The ground truth. Shape should be [b, 1, 256, 256]
        """
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, weight=self.weight, reduction=self.reduction)

        # Dice Loss
        # Flatten the tensors
        smooth = 1e-6
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        intersection = (logits_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (logits_flat.sum() + targets_flat.sum() + smooth)

        # Total loss: BCE + Dice
        total_loss = bce_loss + dice_loss

        return total_loss


class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(weight)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target)
        loss_dice = self.diceloss(pred, target, softmax=False)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        info_name = os.path.join(log_dir, '{}.info.log'.format(name))
        info_handler = logging.handlers.TimedRotatingFileHandler(
            info_name, when='D', encoding='utf-8'
        )
        info_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        info_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(info_handler)
        logger.addHandler(console_handler)

    return logger


def log_config_info(config, logger):
    log_info = '#----------Config info----------#'
    logger.info(log_info)
    for name, value in inspect.getmembers(config):
        if name.startswith('_') or inspect.isfunction(value) or inspect.ismethod(value):
            continue
        logger.info(f'{name}: {value},')


def build_model(network, model_config):
    if network == 'unet':
        from seg_model.model_zoo.unet.unet import unet
        model = unet(
            in_channels=model_config['input_channels'],
            classes=model_config['num_classes'],
        )
    elif network == 'swinunet':
        from seg_model.model_zoo.swinunet.vision_transformer import SwinUnet
        model = SwinUnet(model_config)
    elif network == 'vmunet':
        from seg_model.model_zoo.vmunet.vmunet import VMUNet
        model = VMUNet(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            depths=model_config['depths'],
            depths_decoder=model_config['depths_decoder'],
            drop_path_rate=model_config['drop_path_rate'],
        )
    elif network == 'ukan':
        from seg_model.model_zoo.ukan.archs import UKAN
        model = UKAN(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            embed_dims=model_config['input_list'],
        )
    elif network == 'unetpp':
        from seg_model.model_zoo.unetpp.unetpp import ResNet34UnetPlus
        model = ResNet34UnetPlus(
            num_class=model_config['num_classes'],
            num_channels=model_config['input_channels'],
        )
    elif network == 'segnet':
        from seg_model.model_zoo.segnet.segnet import SegNet
        model = SegNet(
            label_nbr=model_config['num_classes'],
            input_nbr=model_config['input_channels'],
        )
    elif network == 'segformer':
        from seg_model.model_zoo.segformer.segformer import SegFormer
        model = SegFormer(
            num_classes=model_config['num_classes'],
        )
    elif network == 'hrnet':
        from seg_model.model_zoo.hrnet.hrnet import HighResolutionNet
        model = HighResolutionNet(
            num_classes=model_config['num_classes'],
        )
    elif network == 'lightmunet':
        from seg_model.model_zoo.lightmunet.lightmunet import LightMUNet
        model = LightMUNet(
            out_channels=model_config['num_classes'],
        )
    elif network == 'transunet':
        from seg_model.model_zoo.transunet.transunet import VisionTransformer
        model = VisionTransformer(
            num_classes=model_config['num_classes'],
        )
    elif network == 'unext':
        from seg_model.model_zoo.unext.unext import UNext
        model = UNext(
            num_classes=model_config['num_classes'],
        )
    elif network == 'mcure':
        from seg_model.model_zoo.mcure.mcure import MCURE
        model = MCURE(
            num_classes=model_config['num_classes'],
        )
    return model









