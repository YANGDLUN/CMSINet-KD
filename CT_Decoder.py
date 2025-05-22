import torch
import torch.nn as nn

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
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Transconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CT_decoder(nn.Module):
    def __init__(self, ):
        super(CT_decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder3 = nn.Sequential(
            BasicConv2d(512, 320,3, padding=1 ),
            TransBasicConv2d(320, 320, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )

        self.decoder2 = nn.Sequential(
            BasicConv2d(640, 320,3, padding=1 ),
            BasicConv2d(320, 128,3, padding=1 ),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S1 = nn.Conv2d(64, 41, 3, stride=1, padding=1)

        self.decoder0 = nn.Sequential(
            BasicConv2d(128, 64, 3, padding=1),
        )
        self.S0 = nn.Conv2d(64, 41, 3, stride=1, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x3, x2, x1, x0):
        x3_up = self.decoder3(x3)
        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        x0_up = self.decoder0(torch.cat((x0, x1_up), 1))

        s0 = self.S0(x0_up)
        s0 = self.upsample4(s0)

        return s0,x2_up,x1_up



