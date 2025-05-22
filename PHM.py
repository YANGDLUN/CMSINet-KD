import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from yzyall.yzy.mymodelnew.toolbox.models.SeaNet.model.SeaNet_models import DSConv3x3

class CIR(nn.Module):
    """
    The implementation of RGB-induced details enhancement module.
    """

    def __init__(self, channel: int, k=3) -> None:
        super(CIR, self).__init__()

        self.Avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            DSConv3x3(channel, channel),
            nn.Sigmoid()
        )
        self.CBR = nn.Sequential(
            DSConv3x3(channel,channel,),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv1x3 = nn.Conv2d(channel// 2 , channel// 2 , kernel_size=(1, k),stride=2, padding=(0, k // 2))
        self.conv3x1 = nn.Conv2d(channel// 2 , channel // 2, kernel_size=(k, 1), padding=(k // 2, 0))

        self.conv3x1_2 = nn.Conv2d(channel // 2, channel // 2, kernel_size=(k, 1), stride=2, padding=(k // 2, 0))
        self.conv1x3_2 = nn.Conv2d(channel // 2, channel // 2 , kernel_size=(1, k), padding=(0, k // 2))

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, input_enhance: Tensor, input_rgb: Tensor) -> Tensor:
        rgbd1x3 = self.conv1x3(input_enhance)
        rgbd3x1 = self.conv3x1(rgbd1x3)

        rgbdd3x1 = self.conv3x1_2(input_enhance)
        rgbdd1x3 = self.conv1x3_2(rgbdd3x1)
        rgbd1331 = torch.cat([rgbdd1x3, rgbd3x1], dim=1)
        rgbd1331 = self.CBR(rgbd1331)

        x, _ = torch.max(input_rgb, dim=1, keepdim=True)
        x = self.conv2(self.conv1(x))
        mask_MC = torch.sigmoid(x)  ###1xHxW

        x_globalAve = self.Avg(input_rgb)  ###1xHxW

        fuse = rgbd1331 * mask_MC + rgbd1331 * x_globalAve
        rgb_enhance = fuse + input_rgb

        return rgb_enhance