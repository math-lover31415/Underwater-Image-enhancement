import torch
import torch.nn.functional as F
from constants import DCP_TRANSMISSION_MAP_SCALING, DCP_KERNEL_SIZE, DCP_TRANSMISSION_PERCENTILE, DCP_MIN_TRANSMISSION, ATTENTION_SCALING
import cv2
import numpy as np

def apply_clahe_rgb(img, clipLimit=2.0, tileGridSize=(8,8)):
    img_uint8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb_clahe.astype(np.float32) / 255.0

def minPool2d(x, kernel_size, stride=None, padding=0):
    return -F.max_pool2d(-x, kernel_size, stride=stride, padding=padding)

def backgroundLight(x, dark_channel):
    B, C, H, W = x.shape
    n_top = max(1, int(H * W * DCP_TRANSMISSION_PERCENTILE))
    dc_flat = dark_channel.view(B, -1)
    _, top_idx = dc_flat.topk(n_top, dim=1)
    img_flat = x.view(B, C, -1)
    idx_expanded = top_idx.unsqueeze(1).expand(B, C, n_top)
    top_pixels = img_flat.gather(dim=2, index=idx_expanded)
    A = top_pixels.max(dim=2).values.view(B, C, 1, 1)
    return A

def DCPTransmission(x):
    min_channel, _ = x[:, 1:, :, :].min(dim=1, keepdim=True)
    dark_channel = minPool2d(min_channel, kernel_size=DCP_KERNEL_SIZE, stride=1, padding=(DCP_KERNEL_SIZE-1)//2)
    A = backgroundLight(x, dark_channel)
    transmission_map = 1 - DCP_TRANSMISSION_MAP_SCALING * dark_channel
    return transmission_map, A

def enhanceDCP(x):
    batched = x.ndim==4
    if not batched:
        x = x.unsqueeze(0)
    t, A = DCPTransmission(x)
    J = (x-A)/torch.clamp(t, DCP_MIN_TRANSMISSION, 1) + A
    if not batched:
        J = J.squeeze(0)
    return J

def reverseTransmissionMap(x):
    return 1 - ATTENTION_SCALING*DCPTransmission(x)[0]

def applyMapBasedAttention(x,map):
    resizedAttentionMap = F.interpolate(map, size=x.shape[2:], mode='bilinear', align_corners=False)
    x = x * resizedAttentionMap
    return x


if __name__ == "__main__":
    image = torch.rand(3,256,256)
    enhanced = enhanceDCP(image)
    print(enhanced.shape)