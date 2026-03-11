import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models

from model import ImageEnhancementNetwork
from constants.TrainingParameters import *


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        mobile = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # First 7 features ≈ relu3 equivalent, much lighter than VGG
        self.feature_extractor = nn.Sequential(*list(mobile.features)[:7]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        del mobile

    def forward(self, output, gt):
        feat_out = self.feature_extractor(output)
        feat_gt  = self.feature_extractor(gt)
        return F.mse_loss(feat_out, feat_gt)


class CompositeLoss(nn.Module):
    def __init__(self, model: ImageEnhancementNetwork, lambda_mae=100, lambda_ssim=40, lambda_hf=30, lambda_lf=60, lambda_perceptual=2, lambda_color=60):
        super(CompositeLoss, self).__init__()
        self.lambda_mae = lambda_mae
        self.lambda_ssim = lambda_ssim
        self.lambda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.lambda_color = lambda_color
        self.lambda_perceptual = lambda_perceptual
        self.model = model
        
        self.mae_refinement = nn.L1Loss()
        self.mse_prelim = nn.MSELoss()
        self.l1_grad = nn.L1Loss()
        self.l1_prelim = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        
    
    def color_loss(self, output, gt):
        # Penalize mean color deviation per channel
        rgb_loss = F.l1_loss(output.mean(dim=[2, 3]), gt.mean(dim=[2, 3]))

        luminance_out = 0.299*output[:,0:1] + 0.587*output[:,1:2] + 0.114*output[:,2:3]
        luminance_gt  = 0.299*gt[:,0:1]     + 0.587*gt[:,1:2]     + 0.114*gt[:,2:3]
        chroma_out = output - luminance_out
        chroma_gt  = gt - luminance_gt
        
        return rgb_loss + 2.0 * F.l1_loss(chroma_out, chroma_gt)

    def ssim_loss(self, output, gt):
        return 1 - ssim(output, gt, data_range=1.0, size_average=True)
    
    
    def forward(self, lf, hf, gt_lf, gt_hf):
        gt = gt_hf + gt_lf
        output, lf_output, hf_output = self.model(lf,hf)

        l_mae_refinement = self.mae_refinement(output, gt)
        l_ssim = self.ssim_loss(output, gt)
        l_perceptual = torch.log1p(self.perceptual(output, gt))
        l_color = self.color_loss(output, gt)
        
        l_mse_prelim = self.mse_prelim(lf_output,gt_lf)

        l_mae_prelim = self.l1_prelim(hf_output, gt_hf)
        
        l_refinement = self.lambda_mae * l_mae_refinement + self.lambda_ssim * l_ssim + self.lambda_perceptual*l_perceptual + l_color*self.lambda_color

        l_prelim = self.lambda_lf*l_mse_prelim + self.lambda_hf*l_mae_prelim

        return l_refinement + l_prelim