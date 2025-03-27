import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

FEATURE_MAPS = {}


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        return x


class AdaptiveDeepSeparableConvolutionalFrequencyFilters(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):
        return self.depthwise_separable_conv(x)


class AdaptiveResidualFrequencyFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):
        return x - self.depthwise_separable_conv(x)


class OffsetGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 18, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class FreqFusion(nn.Module):
    def __init__(self, high_channels, low_channels, output_channels, drop_path_rate=0.1, Hw=1.45, Lw=0.85):
        super().__init__()

        self.ADFF = AdaptiveDeepSeparableConvolutionalFrequencyFilters(high_channels)
        self.offset = OffsetGenerator(high_channels)
        self.ARFF = AdaptiveResidualFrequencyFilter(low_channels)
        self.deform_conv = DeformConv2d(high_channels, output_channels, kernel_size=3,
                                        padding=1)  # Placeholder for DeformConv2d
        self.drop_path = nn.Identity()

        self.norm1 = nn.LayerNorm(output_channels)
        self.norm2 = nn.LayerNorm(output_channels)
        self.mlp = nn.Sequential(
            nn.Linear(output_channels, 4 * output_channels),
            nn.GELU(),
            nn.Linear(4 * output_channels, output_channels)
        )
        self.Hw = Hw
        self.Lw = Lw
        self.low_level_conv = nn.Conv2d(low_channels, output_channels, kernel_size=1)

    def forward(self, high_level, low_level):
        orgin_level_up = F.interpolate(high_level, size=low_level.shape[2:], mode='bilinear', align_corners=False)

        # ADFF 提取
        orgin_level_smooth = self.ADFF(orgin_level_up)

        # 保存 ADFF 特征图
        FEATURE_MAPS['ADFF'] = orgin_level_smooth.detach().cpu()

        a = self.Hw
        b = self.Lw
        offsets = self.offset(orgin_level_smooth)
        orgin_level_refined = self.deform_conv(orgin_level_smooth, offsets)

        # ARFF 提取
        Residual_level_detail = self.ARFF(low_level)

        # 保存 ARFF 特征图
        FEATURE_MAPS['ARFF'] = Residual_level_detail.detach().cpu()

        Residual_level_detail = self.low_level_conv(Residual_level_detail)

        fused = orgin_level_refined * b + Residual_level_detail * a

        B, C, H, W = fused.shape
        fused_norm = self.norm1(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fused_mlp = self.mlp(fused_norm.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        fused = fused + self.drop_path(fused_mlp)

        fused_norm = self.norm2(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fused = fused + self.drop_path(fused_norm)

        return fused


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 轻量级通道注意力（ECA风格）
class ECASelector(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.selector = ECASelector(out_channels)
        self.freq_fusion = FreqFusion(out_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        selector_weights = self.selector(x)
        x = self.freq_fusion(x * selector_weights, x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetWithFreqFECASelect_deepconv(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True ):
        super(UNetWithFreqFECASelect_deepconv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

