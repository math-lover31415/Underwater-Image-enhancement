import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from dct import splitImage 
from constants import DCT_CUTOFF_RATIO
from constants.TrainingParameters import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processImage(img_path, target_size=(256, 256), split=True, multiChannel=True):
    image = Image.open(img_path).convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    img_np = np.array(image) / 255.0

    if split:
        hf_rgb, lf_rgb = splitImage(img_np, cutoff_ratio=DCT_CUTOFF_RATIO)
        hf_rgb = np.clip(hf_rgb, 0, 1)
        lf_rgb = np.clip(lf_rgb, 0, 1)

        if multiChannel:
            img_uint8 = (img_np * 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] /= 180.0
            hsv[:, :, 1] /= 255.0
            hsv[:, :, 2] /= 255.0
            
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
    
def splitTensor(img: torch.Tensor):
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
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.target_size = target_size
        self.emulatedFunction = emulatedFunction
        
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

    def apply_physics_augmentation(self, gt_tensor):
        C, H, W = gt_tensor.shape
        beta_r = random.uniform(0.15, 0.9)
        beta_g = random.uniform(0.05, 0.3)
        beta_b = random.uniform(0.05, 0.3)
        beta = torch.tensor([beta_r, beta_g, beta_b]).view(3, 1, 1)
        
        d = torch.rand(1, H, W)
        d = TF.gaussian_blur(d, kernel_size=[31, 31], sigma=[5.0, 5.0])
        d = d / d.max()
        
        t = torch.exp(-beta * d)
        
        A_r = random.uniform(0.6, 1.0)
        A_g = random.uniform(0.6, 1.0)
        A_b = random.uniform(0.8, 1.0)
        A = torch.tensor([A_r, A_g, A_b]).view(3, 1, 1)
        
        I = gt_tensor * t + A * (1 - t)
        return torch.clamp(I, 0.0, 1.0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        input_path = os.path.join(self.input_dir, fname)
        
        if self.emulatedFunction is None:
            gt_path = os.path.join(self.gt_dir, fname)
            
            if random.random() < 0.3:
                gt_image = Image.open(gt_path).convert('RGB').resize(self.target_size, Image.Resampling.LANCZOS)
                gt_tensor = TF.to_tensor(gt_image)
                syn_input = self.apply_physics_augmentation(gt_tensor)
                
                img_np = syn_input.permute(1, 2, 0).numpy()
                hf_rgb, lf_rgb = splitImage(img_np, cutoff_ratio=DCT_CUTOFF_RATIO)
                hf_rgb = np.clip(hf_rgb, 0, 1)
                lf_rgb = np.clip(lf_rgb, 0, 1)
                
                img_uint8 = (img_np * 255).astype(np.uint8)
                lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
                hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] /= 180.0
                hsv[:, :, 1] /= 255.0
                hsv[:, :, 2] /= 255.0
                
                lf_np = np.concatenate([lf_rgb, lab, hsv], axis=-1)
                hf_np = np.concatenate([hf_rgb, lab, hsv], axis=-1)
                
                lf = torch.from_numpy(lf_np).permute(2, 0, 1).float()
                hf = torch.from_numpy(hf_np).permute(2, 0, 1).float()
                gt_lf, gt_hf = processImage(gt_path, self.target_size, multiChannel=False)
            else:
                lf, hf = processImage(input_path, self.target_size, multiChannel=True)
                gt_lf, gt_hf = processImage(gt_path, self.target_size, multiChannel=False)
        else:
            lf, hf = processImage(input_path, self.target_size, multiChannel=True)
            gt = self.emulatedFunction(lf+hf)
            gt_lf, gt_hf = splitTensor(gt)

        # Standard Augmentations
        if random.random() < 0.5:
            lf = TF.hflip(lf); hf = TF.hflip(hf)
            gt_lf = TF.hflip(gt_lf); gt_hf = TF.hflip(gt_hf)
            
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            lf = torch.rot90(lf, k, [1, 2]); hf = torch.rot90(hf, k, [1, 2])
            gt_lf = torch.rot90(gt_lf, k, [1, 2]); gt_hf = torch.rot90(gt_hf, k, [1, 2])
            
        b_jitter = random.uniform(-0.1, 0.1)
        lf[:3] = torch.clamp(lf[:3] + b_jitter, 0.0, 1.0)
        hf[:3] = torch.clamp(hf[:3] + b_jitter, 0.0, 1.0)
        gt_lf[:3] = torch.clamp(gt_lf[:3] + b_jitter, 0.0, 1.0)
        gt_hf[:3] = torch.clamp(gt_hf[:3] + b_jitter, 0.0, 1.0)

        return lf, hf, gt_lf, gt_hf
    
if __name__=='__main__':
    data = ImageDataset(TEST_DATA_PATH+"/input", TEST_DATA_PATH+"/GT", limitImages=10)
    loader = DataLoader(data, batch_size=1)
    for i in loader:
        for x in i:
            print(x.shape)
        break