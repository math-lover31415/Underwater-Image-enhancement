import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from constants import DCT_CUTOFF_RATIO

def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def splitImage(img, cutoff_ratio=0.2):
    h, w, c = img.shape
    
    High = np.zeros_like(img)
    Low = np.zeros_like(img)
    
    cutoff = int(cutoff_ratio * min(h, w))

    for ch in range(3):
        Orig = img[:, :, ch]
        Orig_T = dct2(Orig)

        low_mask = np.zeros_like(Orig_T)
        low_mask[:cutoff, :cutoff] = 1
        high_mask = 1 - low_mask

        Low_T = Orig_T * low_mask
        High_T = Orig_T * high_mask
        
        High[:, :, ch] = idct2(High_T)
        Low[:, :, ch]  = idct2(Low_T)
    return High, Low


if __name__ == "__main__":
    input_dir = "data/DCT/Inputs"
    hf_dir = "data/DCT/HF"
    lf_dir = "data/DCT/LF"

    os.makedirs(hf_dir, exist_ok=True)
    os.makedirs(lf_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        img = np.array(Image.open(path).convert("RGB")) / 255.0
        High, Low = splitImage(img, cutoff_ratio=DCT_CUTOFF_RATIO)
        Image.fromarray(np.clip(High*255,0,255).astype(np.uint8)).save(
            os.path.join(hf_dir, fname))
        
        Image.fromarray(np.clip(Low*255,0,255).astype(np.uint8)).save(
            os.path.join(lf_dir, fname))
