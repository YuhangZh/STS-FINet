import torch.nn as nn
import torch
from .modules import DiffFusionBlock, MixCfn
from .segformer_head import segformer_head
from .CSWin_transformer2 import CSWin_De
from torchvision import models
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, backbone='resnet18'):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Encoder, self).__init__()
        if backbone == 'resnet18':
            self.resnet = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()

        for n, m in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.head1 = MixCfn(in_features=256, out_features=128)
        self.head2 = MixCfn(in_features=512, out_features=128)
                                  
    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)       # 1/2, in=3, out=64
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x_4 = self.resnet.maxpool(x)   # 1/4, in=64, out=64
        x_4 = self.resnet.layer1(x_4)  # 1/4, in=64, out=64

        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        x_16 = self.resnet.layer3(x_8)  # 1/16, in=128, out=256
        x_16_ = self.head1(x_16)

        x_32 = self.resnet.layer4(x_16)  # 1/32, in=256, out=512
        x_32_ = self.head2(x_32)

        return x_4, x_8, x_16_, x_32_

    def forward(self, img1, img2):
        x1 = self.forward_single(img1)
        x2 = self.forward_single(img2)

        return x1, x2


class Decoder(nn.Module):
    def __init__(self, img_size=512, num_classes=7):
        super(Decoder, self).__init__()
        # Decoder
        self.seg_chans = (64, 128, 128, 128)
        self.img_size = img_size
        self.CSWin = nn.ModuleList([CSWin_De(img_size=self.img_size//4, in_chans=self.seg_chans[0]*3,
                                             embed_dim=self.seg_chans[0]*3, depth=2),
                                    CSWin_De(img_size=self.img_size//8, in_chans=self.seg_chans[1]*3,
                                             embed_dim=self.seg_chans[1]*3, depth=2),
                                    CSWin_De(img_size=self.img_size//8, in_chans=self.seg_chans[2]*3,
                                             embed_dim=self.seg_chans[2]*3, depth=2),
                                    CSWin_De(img_size=self.img_size//8, in_chans=self.seg_chans[3]*3,
                                             embed_dim=self.seg_chans[3]*3, depth=2)])

        # CNNDecoder
        self.up_change = segformer_head(in_channels=self.seg_chans, embed_dim=self.seg_chans[0], num_classes=1)
        self.Diff_fusions = nn.ModuleList()
        for i_layer in range(len(self.seg_chans)):
            layer = DiffFusionBlock(in_chanel=self.seg_chans[i_layer])
            self.Diff_fusions.append(layer)

        # CSWinDecoder
        self.up_CSWin = segformer_head(in_channels=self.seg_chans, embed_dim=self.seg_chans[0], num_classes=num_classes)

    def forward(self, x1, x2):
        out = []
        x_lcm1, x_lcm2, x_change = [], [], []

        # CNNDecoder
        for i in range(len(x1)):
            x = self.Diff_fusions[i](x1[i], x2[i])
            out.append(x)

        # CSWinDecoder
        for i in range(len(x1)):
            x = torch.cat([x1[i], x2[i], out[i]], 1)
            x = self.CSWin[i](x)
            lcm1 = x[:, 0:self.seg_chans[i], :, :]
            lcm2 = x[:, self.seg_chans[i]:self.seg_chans[i]*2, :, :]
            change = x[:, self.seg_chans[i]*2:, :, :]
            x_lcm1.append(lcm1)
            x_lcm2.append(lcm2)
            x_change.append(change)

        x = self.up_change(x_change)
        xa = self.up_CSWin(x_lcm1)
        xb = self.up_CSWin(x_lcm2)

        return x, xa, xb


class MCDNet(nn.Module):
    def __init__(self, num_classes=7, img_size=512):
        super(MCDNet, self).__init__()

        self.Encoder = Encoder(backbone='resnet34')

        self.Decoder = Decoder(img_size=img_size, num_classes=num_classes)

    def forward(self, img1, img2):
        image_hw = img1.shape[2:]

        x1, x2 = self.Encoder(img1, img2)

        change, out1, out2 = self.Decoder(x1, x2)

        return F.interpolate(change, size=image_hw, mode='bilinear', align_corners=False), \
            F.interpolate(out1, size=image_hw, mode='bilinear', align_corners=False),\
            F.interpolate(out2, size=image_hw, mode='bilinear', align_corners=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
