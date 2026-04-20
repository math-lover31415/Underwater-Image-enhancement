import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models

from model import ImageEnhancementNetwork
from constants.TrainingParameters import *

import kornia.color as KC


class CompositeLoss(nn.Module):
    def __init__(self, model: ImageEnhancementNetwork,
                 lambda_mae=100, lambda_ssim=40,
                 lambda_hf=60, lambda_lf=60,
                 lambda_color=60, lambda_contrast=40):
        super(CompositeLoss, self).__init__()

        self.lambda_mae = lambda_mae
        self.lambda_ssim = lambda_ssim
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.lambda_color = lambda_color
        self.lambda_contrast = lambda_contrast

        self.model = model
        self.mae_refinement = nn.L1Loss()
        self.mse_prelim = nn.MSELoss()
        self.l1_prelim = nn.L1Loss()

    def ssim_loss(self, output, gt):
        return 1 - ssim(output, gt, data_range=1.0, size_average=True)

    def lab_color_loss(self, output, gt):
        lab_out = KC.rgb_to_lab(output)
        with torch.no_grad():
            lab_gt = KC.rgb_to_lab(gt)
        ab_out = lab_out[:, 1:, :, :]
        ab_gt  = lab_gt[:, 1:, :, :]
        return F.l1_loss(ab_out, ab_gt, reduction='mean')

    def local_contrast_loss(self, output, gt):
        gray_out = output.mean(dim=1, keepdim=True)
        with torch.no_grad():
            gray_gt = gt.mean(dim=1, keepdim=True)

        def local_std(gray):
            mean   = F.avg_pool2d(gray, kernel_size=8, stride=1, padding=3)
            mean_sq = F.avg_pool2d(gray * gray, kernel_size=8, stride=1, padding=3)
            var    = mean_sq - mean * mean
            std    = torch.sqrt(var.clamp(min=1e-8))
            return std

        std_out = local_std(gray_out)
        with torch.no_grad():
            std_gt = local_std(gray_gt)
        return F.l1_loss(std_out, std_gt, reduction='mean')

    def forward(self, lf, hf, gt_lf, gt_hf):
        gt = gt_hf + gt_lf
        output, lf_output, hf_output = self.model(lf, hf)

        small_out = F.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)
        small_gt  = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False)

        l_mae_refinement = self.mae_refinement(output, gt)
        l_ssim = self.ssim_loss(output, gt)
        l_color = self.lab_color_loss(small_out, small_gt)
        l_contrast = self.local_contrast_loss(small_out, small_gt)
        
        # Frequency domain loss (luminance channel FFT magnitude)
        lum_out = output.mean(dim=1, keepdim=True)
        lum_gt  = gt.mean(dim=1, keepdim=True)
        fft_out = torch.fft.rfft2(lum_out).abs()
        with torch.no_grad():
            fft_gt = torch.fft.rfft2(lum_gt).abs()
        l_freq = F.l1_loss(fft_out, fft_gt)

        l_mse_prelim = self.mse_prelim(lf_output, gt_lf)
        l_mae_prelim = self.l1_prelim(hf_output, gt_hf)

        l_refinement = (
            self.lambda_mae * l_mae_refinement +
            self.lambda_ssim * l_ssim +
            self.lambda_color * l_color +
            self.lambda_contrast * l_contrast +
            20 * l_freq
        )

        l_prelim = (
            self.lambda_lf * l_mse_prelim +
            self.lambda_hf * l_mae_prelim
        )

        return l_refinement + l_prelim