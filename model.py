import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=23):
        super().__init__()
        f = [64, 128, 256, 512]

        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for feat in f:
            self.encoders.append(DoubleConv(ch, feat))
            self.pools.append(nn.MaxPool2d(2))
            ch = feat

        self.bottleneck = DoubleConv(f[-1], f[-1]*2)

        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = f[-1]*2
        for feat in reversed(f):
            self.upconvs.append(nn.ConvTranspose2d(ch, feat, 2, stride=2))
            self.decoders.append(DoubleConv(feat*2, feat))
            ch = feat

        self.final = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        skips, out = [], x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = up(out)
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        return self.final(out)