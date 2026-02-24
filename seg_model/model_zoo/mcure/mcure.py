from seg_model.model_zoo.mcure.vmamba import VSSM
# from vmamba import VSSM # debug use
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def upsize(x, scale_factor=2, mode='nearest'):
    x = F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False, groups=32)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        '''
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2,**up_kwargs)
        '''

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=32)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DilationDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False, groups=out_channels // 2)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.dila_1 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=out_channels // 2)
        self.dila_2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, groups=out_channels // 2)
        self.dila_3 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=4, dilation=4, bias=False, groups=out_channels // 2)

        self.conv3 = nn.Conv2d((out_channels // 2) * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels // 2)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x_1 = self.dila_1(x)
        x_2 = self.dila_2(x)
        x_3 = self.dila_3(x)
        x = self.conv3(torch.cat([x_1, x_2, x_3], 1))
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class MCURE(nn.Module):
    def __init__(self, 
                 # input_channels=3,
                 # num_classes=1,
                 # mid_channel = 48,
                 # depths=[2, 2, 9, 2],
                 # depths_decoder=[2, 9, 2, 2],
                 # drop_path_rate=0.2,
                 # load_ckpt_path=None,
                 input_channels=3,
                 num_classes=1,
                 mid_channel=48,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )

        # pp parameter
        self.mix = nn.Parameter(torch.FloatTensor(4))
        self.mix.data.fill_(1)
        resnet = models.resnet34(pretrained=False)

        # pp encoder
        self.firstconv = BasicConv2d(input_channels, 64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.vm_conv1 = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.vm_conv2 = nn.ConvTranspose2d(192, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.vm_conv3 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.vm_conv4 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deep_conv3 = DilationDecoderBlock(128 + 512, 64)
        self.deep_conv4 = DilationDecoderBlock(256 + 1024, 64)

        self.decoder0_1 = DecoderBlock(in_channels=64 + 128, out_channels=64)
        self.decoder0_2 = DecoderBlock(in_channels=64 + 64 + 256, out_channels=64)
        self.decoder0_3 = DecoderBlock(in_channels=64 + 64 + 64 + 64, out_channels=64)
        self.decoder0_4 = DecoderBlock(in_channels=64 + 64 + 64 + 64 + 64, out_channels=64)

        self.logit1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.logit2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.logit3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.logit4 = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x):
        if x.size()[1] == 1: # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1,3,1,1)
        f1, f2, f3, f4 = self.vmunet(x)     # [b h w c]
        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2)     # f1 [b, 96, 64, 64]
        f1 = self.vm_conv1(f1)          # [b, 64, 128, 128]
        f2 = f2.permute(0, 3, 1, 2)     # f2 [b, 192, 32, 32]
        f2 = self.vm_conv2(f2)          # [b, 128, 64, 64]
        f3 = f3.permute(0, 3, 1, 2)     # f3 [b, 384, 16, 16]
        f3 = self.vm_conv3(f3)          # [b, 256, 32, 32]
        f4 = f4.permute(0, 3, 1, 2)     # f4 [b, 768, 8, 8]
        f4 = self.vm_conv4(f4)          # [b, 512, 16, 16]

        x_ = self.firstconv(x)                        # x [b, 64, 256, 256]
        x = self.maxpool(x_)                          # x [b, 64, 128, 128]
        e1 = self.encoder1(x)                         # x [b, 64, 128, 128]
        e2 = self.encoder2(e1)                        # x [b, 128, 64, 64]
        e3 = self.encoder3(e2)                        # x [b, 256, 32, 32]
        e4 = self.encoder4(e3)                        # x [b, 512, 16, 16]

        x0_0 = x_   # [b, 64, 256, 256]
        x1_0 = torch.cat([e1, f1], 1)   # [b, 128, 128, 128]
        x0_1 = self.decoder0_1([x0_0, upsize(x1_0)])    # 64

        x2_0 = torch.cat([e2, f2], 1)   # 256
        x0_2 = self.decoder0_2([x0_0, x0_1, upsize(x2_0, scale_factor=4)])  # 64

        x3_0 = torch.cat([e3, f3], 1)   # 512
        x3_0 = self.deep_conv3([x1_0, upsize(x3_0, scale_factor=4)])   # 64
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, upsize(x3_0)])    # 64

        x4_0 = torch.cat([e4, f4], 1)   # 1024
        x4_0 = self.deep_conv4([x2_0, upsize(x4_0, scale_factor=4)])   # 64
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, upsize(x4_0, scale_factor=4)])  # 64

        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)

        # print(self.mix)
        logit = self.mix[0] * logit1 + self.mix[1] * logit2 + self.mix[2] * logit3 + self.mix[3] * logit4
        # logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)
        return logit

    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数 
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")


if __name__ == '__main__':
    model = VMUNet_1().cuda()
    model.load_from()
    x = torch.randn(2, 3, 256, 256).cuda()
    predict = model(x)
    # print(predict.shape)  #  deep_supervision true   predict[0] [2, 1, 256, 256] , predict[1] [2, 1, 128, 128] 这两项用于监督



