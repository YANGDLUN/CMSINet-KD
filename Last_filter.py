import torch
import numpy as np
import cv2


# 定义傅里叶频域滤波函数
def apply_frequency_filter(feature_map, filter_type='low_pass', cutoff_freq=0.05):
    # 遍历特征图的每个通道
    filtered_feature_map = torch.zeros_like(feature_map)
    for i in range(feature_map.shape[1]):  # 遍历通道数
        channel = feature_map[:, i, :, :]  # 获取当前通道的特征图

        # 将张量转换为 NumPy 数组，并转换为灰度图像
        channel_numpy = channel.detach().cpu().numpy()[0]
        channel_numpy = np.abs(channel_numpy)  # 取绝对值
        channel_numpy = np.uint8(255 * (channel_numpy - np.min(channel_numpy)) / np.ptp(channel_numpy))

        # 傅里叶变换
        f_transform = np.fft.fft2(channel_numpy)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = channel_numpy.shape
        crow, ccol = rows // 2, cols // 2

        # 生成频域滤波器
        mask = np.zeros((rows, cols), np.uint8)
        if filter_type == 'high_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 1
        elif filter_type == 'low_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 0
            mask = 1 - mask

        # 应用滤波器
        f_shift = f_shift * mask

        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 将 NumPy 数组转换回张量，并复制到结果特征图中的对应通道
        filtered_feature_map[:, i, :, :] = torch.from_numpy(img_back).unsqueeze(0)

    return filtered_feature_map

