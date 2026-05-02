import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.fftpack import dct, idct
from constants import DCT_CUTOFF_RATIO
from constants.TrainingParameters import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def snap_to_multiple(x, multiple_of=16, min_value=16):
    snapped = (x // multiple_of) * multiple_of
    return max(min_value, snapped)

def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def split_image_dct(img, cutoff_ratio=0.2):
    h, w, _ = img.shape
    high = np.zeros_like(img, dtype=np.float32)
    low = np.zeros_like(img, dtype=np.float32)
    cutoff = int(cutoff_ratio * min(h, w))

    for ch in range(3):
        freq = dct2(img[:, :, ch])
        low_mask = np.zeros_like(freq)
        low_mask[:cutoff, :cutoff] = 1.0
        high_mask = 1.0 - low_mask

        low[:, :, ch] = idct2(freq * low_mask)
        high[:, :, ch] = idct2(freq * high_mask)

    return np.clip(high, 0.0, 1.0), np.clip(low, 0.0, 1.0)

def _resize_image(image, target_size=None, multiple_of=16):
    if target_size is not None:
        return image.resize(target_size, Image.Resampling.LANCZOS)

    width, height = image.size
    new_width = snap_to_multiple(width, multiple_of)
    new_height = snap_to_multiple(height, multiple_of)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def _pack_multichannel_tensors(img_np, low_rgb, high_rgb):
    img_uint8 = (img_np * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    lf_np = np.concatenate([low_rgb, lab, hsv], axis=-1)
    hf_np = np.concatenate([high_rgb, lab, hsv], axis=-1)

    lf_tensor = torch.from_numpy(lf_np).permute(2, 0, 1).float()
    hf_tensor = torch.from_numpy(hf_np).permute(2, 0, 1).float()
    return lf_tensor, hf_tensor

def processImage(img_path, target_size=None, split=True, multiChannel=True, multiple_of=16):
    """
    Standardizes image size and performs frequency decomposition.
    """
    image = Image.open(img_path).convert('RGB')
    image = _resize_image(image, target_size=target_size, multiple_of=multiple_of)
    img_np = np.array(image) / 255.0

    if split:
        hf_rgb, lf_rgb = split_image_dct(img_np, cutoff_ratio=DCT_CUTOFF_RATIO)

        if multiChannel:
            return _pack_multichannel_tensors(img_np, lf_rgb, hf_rgb)
        else:
            return (
                torch.from_numpy(lf_rgb).permute(2, 0, 1).float(),
                torch.from_numpy(hf_rgb).permute(2, 0, 1).float(),
            )

    if multiChannel:
        img_uint8 = (img_np * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 180.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        img_np = np.concatenate([img_np, lab, hsv], axis=-1)

        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    return torch.from_numpy(img_np).permute(2, 0, 1).float()
    
def splitTensor(img: torch.Tensor, multiChannel=True):
    img = img.permute(1, 2, 0)
    np_arr = img.cpu().numpy()
    hf, lf = split_image_dct(np_arr, cutoff_ratio=DCT_CUTOFF_RATIO)

    if multiChannel:
        return _pack_multichannel_tensors(np_arr, lf, hf)

    hf = torch.from_numpy(hf).permute(2, 0, 1).float()
    lf = torch.from_numpy(lf).permute(2, 0, 1).float()

    return lf, hf

class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, no_gt_dir=None, target_size=(256, 256), emulatedFunction = None, limitImages = None):
        """
        Args:
            input_dir: Path to degraded images. [cite: 41]
            gt_dir: Path to clean (Ground Truth) images.
            target_size: Standard dimensions for processing. [cite: 183]
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.target_size = target_size
        self.emulatedFunction = emulatedFunction
        
        # We find all images in the input directory 
        try:
            self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except FileNotFoundError:
            self.image_files = []
        
        if emulatedFunction is None and no_gt_dir is not None:
            try:
                no_gt_images = [f for f in os.listdir(no_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            except FileNotFoundError:
                no_gt_images = []
            self.image_files.extend(no_gt_images)
        
        if limitImages is not None:
            self.image_files = self.image_files[:limitImages]

        # Standardization tool kept for backwards compatibility.
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Get the shared filename
        fname = self.image_files[idx]
        
        # 2. Process Input (Decompose into LF/HF) [cite: 13, 64, 186]
        input_path = os.path.join(self.input_dir, fname)
        lf, hf = processImage(input_path, target_size=None, multiChannel=True)
        
        if self.emulatedFunction is None:
            # 3. Process Ground Truth (same preprocessing path as input)
            gt_path = os.path.join(self.gt_dir, fname)
            gt_lf, gt_hf = processImage(gt_path, target_size=None, multiChannel=True)
        else:
            gt = self.emulatedFunction(lf+hf)
            gt_lf, gt_hf = splitTensor(gt, multiChannel=True)
        
        # Return the tensors for train.py
        return lf, hf, gt_lf, gt_hf
    
if __name__=='__main__':
    data = ImageDataset(TEST_DATA_PATH+"/input", TEST_DATA_PATH+"/GT", limitImages=10)
    loader = DataLoader(data, batch_size=1)
    for i in loader:
        for x in i:
            print(x.shape)
        break