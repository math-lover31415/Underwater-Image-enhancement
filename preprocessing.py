import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dct import splitImage 
from constants import DCT_CUTOFF_RATIO
from constants.TrainingParameters import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processImage(img_path, target_size=(256, 256), split=True, multiChannel=True):
    """
    Standardizes image size and performs Frequency Decomposition.
    """
    image = Image.open(img_path).convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    img_np = np.array(image) / 255.0

    # Split frequency FIRST on RGB only
    if split:
        hf_rgb, lf_rgb = splitImage(img_np, cutoff_ratio=DCT_CUTOFF_RATIO)
        hf_rgb = np.clip(hf_rgb, 0, 1)
        lf_rgb = np.clip(lf_rgb, 0, 1)

        if multiChannel:
            img_uint8 = (img_np * 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] /= 180.0   # H: [0,180] → [0,1]
            hsv[:, :, 1] /= 255.0   # S: [0,255] → [0,1]
            hsv[:, :, 2] /= 255.0   # V: [0,255] → [0,1]
            
            # Append color channels AFTER frequency split
            lf_np = np.concatenate([lf_rgb, lab, hsv], axis=-1)
            hf_np = np.concatenate([hf_rgb, lab, hsv], axis=-1)
        else:
            lf_np, hf_np = lf_rgb, hf_rgb

        lf_tensor = torch.from_numpy(lf_np).permute(2, 0, 1).float()
        hf_tensor = torch.from_numpy(hf_np).permute(2, 0, 1).float()
        return lf_tensor, hf_tensor
    else:
        if multiChannel:
            img_uint8 = (img_np * 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            img_np = np.concatenate([img_np, lab, hsv], axis=-1)
        return torch.from_numpy(img_np).permute(2, 0, 1).float()
    
def splitTensor(img: torch.Tensor): # To be refactored
    img = img.permute(1, 2, 0)
    np_arr = img.cpu().numpy()
    hf, lf = splitImage(np_arr)

    hf = np.clip(hf, 0, 1)
    hf = torch.from_numpy(hf).permute(2,0,1).float()
    
    lf = np.clip(lf, 0, 1)
    lf = torch.from_numpy(lf).permute(2,0,1).float()

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
        lf, hf = processImage(input_path, self.target_size, multiChannel=True)
        
        if self.emulatedFunction is None:
            # 3. Process Ground Truth (Just resize and convert to Tensor)
            gt_path = os.path.join(self.gt_dir, fname)
            gt_lf, gt_hf = processImage(gt_path, self.target_size, multiChannel=False)
        else:
            gt = self.emulatedFunction(lf+hf)
            gt_lf, gt_hf = splitTensor(gt)
        
        # Return the tensors for train.py
        return lf, hf, gt_lf, gt_hf
    
if __name__=='__main__':
    data = ImageDataset(TEST_DATA_PATH+"/input", TEST_DATA_PATH+"/GT", limitImages=10)
    loader = DataLoader(data, batch_size=1)
    for i in loader:
        for x in i:
            print(x.shape)
        break