import torch
import torch.nn as nn
from mydesignmodel.yzy_model.FindTheBestDec.Self_Moudles.filter_dep import apply_dep_filter

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(SpatialWeights, self).__init__()
        self.localattention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(self.dim, self.dim , 1),
            nn.BatchNorm2d(self.dim ),
            nn.ReLU()
        )

        self.globalattention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim, self.dim , 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = x1 + x2
        enhanced_feature_map = apply_dep_filter(x, filter_type='low_pass', cutoff_freq=0.7)
        x_d, _ = torch.max(enhanced_feature_map, dim=1, keepdim=True)
        x_d = self.conv2(self.conv1(x_d))
        mask_d = torch.sigmoid(x_d)  ###1xHxW
        spatial_weights = x1 * mask_d + x2 * mask_d

        return spatial_weights


class CIF(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(CIF, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights * x2
        return out, spatial_weights

