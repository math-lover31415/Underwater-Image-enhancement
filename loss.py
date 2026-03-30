import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models

from model import ImageEnhancementNetwork
from constants.TrainingParameters import *

import kornia.color as KC
# major changes didnt take gradient for ground truth 
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        mobile = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(mobile.features)[:7]).eval()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        del mobile

    def forward(self, output, gt):
        feat_out = self.feature_extractor(output)

        with torch.no_grad(): 
            feat_gt = self.feature_extractor(gt)

        return F.mse_loss(feat_out, feat_gt)


class CompositeLoss(nn.Module):
    def __init__(self, model: ImageEnhancementNetwork,
                 lambda_mae=100, lambda_ssim=40,
                 lambda_hf=60, lambda_lf=60,
                 lambda_perceptual=2,
                 lambda_color=10, lambda_contrast=8):
        super(CompositeLoss, self).__init__()

        self.lambda_mae = lambda_mae
        self.lambda_ssim = lambda_ssim
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.lambda_perceptual = lambda_perceptual
        self.lambda_color = lambda_color
        self.lambda_contrast = lambda_contrast

        self.model = model

        self.mae_refinement = nn.L1Loss()
        self.mse_prelim = nn.MSELoss()
        self.l1_prelim = nn.L1Loss()

        self.perceptual = PerceptualLoss()

    def ssim_loss(self, output, gt):
        return 1 - ssim(output, gt, data_range=1.0, size_average=True)

    def lab_color_loss(self, output, gt):
        lab_out = KC.rgb_to_lab(output) 

        with torch.no_grad():  
            lab_gt = KC.rgb_to_lab(gt)

        ab_out = lab_out[:, 1:, :, :] / 128.0
        ab_gt  = lab_gt[:, 1:, :, :] / 128.0

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

        # Downsample for heavy losses
        small_out = F.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)  
        small_gt  = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Refinement losses
        l_mae_refinement = self.mae_refinement(output, gt)
        l_ssim = self.ssim_loss(output, gt)

        l_perceptual = torch.log1p(torch.clamp(self.perceptual(output, gt), min=1e-6))  # FIX: clamp to avoid NaNs

        l_color = self.lab_color_loss(small_out, small_gt)  # FIX: compute on downsampled images
        l_contrast = self.local_contrast_loss(small_out, small_gt)  # FIX: compute on downsampled images

        # Preliminary losses
        l_mse_prelim = self.mse_prelim(lf_output, gt_lf)
        l_mae_prelim = self.l1_prelim(hf_output, gt_hf)

        # Combine
        l_refinement = (
            self.lambda_mae * l_mae_refinement +
            self.lambda_ssim * l_ssim +
            self.lambda_perceptual * l_perceptual +
            self.lambda_color * l_color +
            self.lambda_contrast * l_contrast
        )

        l_prelim = (
            self.lambda_lf * l_mse_prelim +
            self.lambda_hf * l_mae_prelim
        )

        return l_refinement + l_prelim