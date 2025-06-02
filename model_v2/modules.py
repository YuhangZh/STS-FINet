from torch import nn
import torch
from torch.nn import Module, Conv2d, Parameter, Softmax


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class DiffFusionBlock(nn.Module):
    def __init__(self, in_chanel):
        super(DiffFusionBlock, self).__init__()
        self.conv = conv1x1(in_planes=in_chanel, out_planes=in_chanel)
        self.norm = nn.BatchNorm2d(in_chanel)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)

        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )

        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )
        
        self.SR1 = SR(out_channels // 2)
        self.SR2 = SR(out_channels // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        
        x1 = self.SR1(x1)
        x2 = self.SR2(x2)
        
        x = torch.cat([x1, x2], dim=1)

        return x


class MixCfn(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super().__init__()
        self.head = self._build_head(in_features, out_features)
        self.conv = self._build_mix_conv(in_features, out_features)
        self.norm1 = nn.BatchNorm2d(out_features)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.gelu = nn.GELU()

    def _build_head(self, in_features, out_features):
        return nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)

    def _build_mix_conv(self, in_features, out_features):
        return MixConv2d(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_features,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.norm1(out)

        ide = self.head(x)
        ide = self.norm2(ide)

        out += ide
        out = self.gelu(out)

        return out


class SR(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out
