import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dct import splitImage 
from constants import DCT_CUTOFF_RATIO

def processImage(img_path, target_size=(256, 256), split=True):
    """
    Standardizes image size and performs Frequency Decomposition.
    """
    image = Image.open(img_path).convert('RGB') # [cite: 59, 184]
    image = image.resize(target_size, Image.Resampling.LANCZOS) # [cite: 137, 183]
    img_np = np.array(image) / 255.0

    if split:
        # Split into LF (color/lighting) and HF (edges/texture) 
        hf_np, lf_np = splitImage(img_np, cutoff_ratio=DCT_CUTOFF_RATIO)
        
        hf_np = np.clip(hf_np, 0, 1)
        lf_np = np.clip(lf_np, 0, 1)
        
        # Convert to Tensors (C, H, W) for PyTorch 
        lf_tensor = torch.from_numpy(lf_np).permute(2, 0, 1).float()
        hf_tensor = torch.from_numpy(hf_np).permute(2, 0, 1).float()
        return lf_tensor, hf_tensor
    else:
        img_np = np.clip(img_np)
        return torch.from_numpy(img_np).permute(2,0,1).float()
    
def splitTensor(img: torch.Tensor): # To be refactored
    img = img.permute(1, 2, 0)
    np_arr = img.numpy()
    hf, lf = splitImage(np_arr)

    hf = np.clip(hf, 0, 1)
    hf = torch.from_numpy(hf).permute(2,0,1).float()
    
    lf = np.clip(lf, 0, 1)
    lf = torch.from_numpy(lf).permute(2,0,1).float()

    return lf, hf

class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, no_gt_dir=None, target_size=(256, 256), emulatedFunction = None):
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
            self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
        except FileNotFoundError:
            self.image_files = []
        
        if emulatedFunction is None and no_gt_dir is not None:
            try:
                no_gt_images = [f for f in os.listdir(no_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
            except FileNotFoundError:
                no_gt_images = []
            self.image_files.extend(no_gt_images)
        
        # Standardization tool
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
        lf, hf = processImage(input_path, self.target_size)
        
        if self.emulatedFunction is None:
            # 3. Process Ground Truth (Just resize and convert to Tensor)
            gt_path = os.path.join(self.gt_dir, fname)
            gt_lf, gt_hf = processImage(gt_path, self.target_size)
        else:
            gt = self.emulatedFunction(lf+hf)
            gt_lf, gt_hf = splitTensor(gt)
        
        # Return the tensors for train.py
        return lf, hf, gt_lf, gt_hf