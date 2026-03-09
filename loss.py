import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models

from model import ImageEnhancementNetwork
from constants.TrainingParameters import *


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Use first 16 layers (up to relu3_3)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        feat_out = self.feature_extractor(output)
        feat_gt  = self.feature_extractor(gt)
        return F.mse_loss(feat_out, feat_gt)


class CompositeLoss(nn.Module):
    def __init__(self, model: ImageEnhancementNetwork, lambda_mae=100.0, lambda_ssim=1.0, lambda_hf=10, lambda_lf=100, lambda_perceptual=1):
        super(CompositeLoss, self).__init__()
        self.lambda_mae = lambda_mae
        self.lambda_ssim = lambda_ssim
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.lambda_perceptual = lambda_perceptual
        self.model = model
        
        self.mae_refinement = nn.L1Loss()
        self.mse_prelim = nn.MSELoss()
        self.l1_grad = nn.L1Loss()
        self.l1_prelim = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        
    
    def ssim_loss(self, output, gt):
        return 1 - ssim(output, gt, data_range=1.0, size_average=True)
    
    
    def forward(self, lf, hf, gt_lf, gt_hf):
        gt = gt_hf + gt_lf
        output = self.model(lf,hf)
        lf_output = self.model.preliminaryNetwork.lfEnhancement(lf)
        hf_output = self.model.preliminaryNetwork.hfEnhancement(hf)

        l_mae_refinement = self.mae_refinement(output, gt)
        l_ssim = self.ssim_loss(output, gt)
        l_perceptual = self.perceptual(output, gt)
        
        l_mse_prelim = self.mse_prelim(lf_output,gt_lf)

        l_mae_prelim = self.l1_prelim(hf_output, gt_hf)
        
        l_refinement = self.lambda_mae * l_mae_refinement + self.lambda_ssim * l_ssim + self.lambda_perceptual*l_perceptual

        l_prelim = self.lambda_lf*l_mse_prelim + self.lambda_hf*l_mae_prelim

        return l_refinement + l_prelim