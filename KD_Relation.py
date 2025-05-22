import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class DSC33(nn.Module):
    def __init__(self,channel,k=3 ):
        super(DSC33,self).__init__()

        self.depwis = nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=channel,bias=False)
        self.pointwise = nn.Conv2d(channel,channel,1,bias=False)

    def forward(self,x):
        x = self.depwis(x)
        x = self.pointwise(x)
        return x

class MultiScaleConv(nn.Module):
    def __init__(self, channels):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = DSC33(channels, k=3)
        self.conv5x5 = DSC33(channels, k=5)
        self.conv7x7 = DSC33(channels, k=7)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        conv7x7 = self.conv7x7(x)
        return conv3x3 + conv5x5 + conv7x7

def apply_freq_filter_per_channel(feature_map, filter_type='high_pass', cutoff_freq=0.7):
    feature_map = feature_map.to(dtype=torch.float32)
    f_transform = torch.fft.fft2(feature_map)
    f_shift = torch.fft.fftshift(f_transform)
    _, channels, rows, cols = feature_map.shape
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros_like(feature_map)
    r = int(cutoff_freq * crow)
    c = int(cutoff_freq * ccol)

    if filter_type == 'high_pass':
        mask[:, :, crow-r:crow+r, ccol-c:ccol+c] = 1
    elif filter_type == 'low_pass':
        mask[:, :, crow-r:crow+r, ccol-c:ccol+c] = 0
        mask = 1 - mask

    f_filtered = f_shift * mask
    f_ishift = torch.fft.ifftshift(f_filtered)
    filtered_feature_map = torch.fft.ifft2(f_ishift)
    return torch.abs(filtered_feature_map)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x

class SpatialWeights(nn.Module):
    def __init__(self, dim,outdim, reduction=1):
        self.dim = dim
        super(SpatialWeights, self).__init__()
        self.multi_scale_conv = MultiScaleConv(outdim)
        self.spatial_attention = SpatialAttention()
        self.conv = nn.Conv2d(dim, outdim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1,x2):
        x1 = F.interpolate(input=x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        combined_features = x1 + x2
        multi_scale_features = self.multi_scale_conv(combined_features)
        # filtered_features = apply_freq_filter_per_channel(multi_scale_features, 'low_pass', 0.7)
        attention_features = self.spatial_attention(multi_scale_features)

        return attention_features