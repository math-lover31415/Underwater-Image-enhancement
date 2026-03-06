import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from utilities import enhanceDCP
from model import ImageEnhancementNetwork
from preprocessing import ImageDataset
from constants.TrainingParameters import *


# Configs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CompositeLoss(nn.Module):
    def __init__(self, model: ImageEnhancementNetwork, lambda_mse=100.0, lambda_ssim=1.0, lambda_gradient=5.0, lambda_hf=10, lambda_lf=100):
        super(CompositeLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        self.lambdda_hf = lambda_hf
        self.lambda_lf = lambda_lf
        self.model = model
        
        self.mse_refinement = nn.MSELoss()
        self.mse_prelim = nn.MSELoss()
        self.l1_grad = nn.L1Loss()
        self.l1_prelim = nn.L1Loss()
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def ssim_loss(self, output, gt):
        return 1 - ssim(output, gt, data_range=1.0, size_average=True)
    
    def gradient_loss(self, output, gt):
        # Compare spatial gradients between output and GT for edge sharpness
        grad_loss = 0.0
        for c in range(output.shape[1]):
            out_c = output[:, c:c+1, :, :]
            gt_c = gt[:, c:c+1, :, :]
            
            out_grad_x = F.conv2d(out_c, self.sobel_x, padding=1)
            out_grad_y = F.conv2d(out_c, self.sobel_y, padding=1)
            gt_grad_x = F.conv2d(gt_c, self.sobel_x, padding=1)
            gt_grad_y = F.conv2d(gt_c, self.sobel_y, padding=1)
            
            grad_loss += self.l1_grad(out_grad_x, gt_grad_x) + self.l1_grad(out_grad_y, gt_grad_y)
        
        return grad_loss / output.shape[1]
    
    def forward(self, lf, hf, gt_lf, gt_hf):
        gt = gt_hf + gt_lf
        output = self.model(lf,hf)
        lf_output = self.model.preliminaryNetwork.lfEnhancement(lf)
        hf_output = self.model.preliminaryNetwork.lfEnhancement(hf)

        l_mse_refinement = self.mse_refinement(output, gt)
        l_ssim = self.ssim_loss(output, gt)
        l_gradient = self.gradient_loss(output, gt)
        
        l_mse_prelim = self.mse_prelim(lf_output,gt_lf)

        l_mae_prelim = self.l1_prelim(hf_output, hf)
        
        l_refinement = self.lambda_mse * l_mse_refinement + self.lambda_ssim * l_ssim + self.lambda_gradient * l_gradient

        l_prelim = self.lambda_lf*l_mse_prelim + self.lambdda_hf*l_mae_prelim

        return l_refinement + l_prelim


class UnderwaterTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, numEpochs, modelSavePoint):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = DEVICE
        self.numEpochs = numEpochs
        self.savePoint = modelSavePoint
        self.loss = CompositeLoss(model)

        self.best_val_loss = float('inf')

        
    def save_model(self):
        if self.savePoint:
            torch.save(self.model.state_dict(), self.savePoint)
    
    def load_model(self):
        if self.savePoint:
            self.model.load_state_dict(torch.load(self.savePoint, map_location=DEVICE, weights_only=False))

    def train_one_epoch(self) -> float:
        self.model.train()
        self.loss.train()
        running_loss = 0.0
        
        for batch_idx, (lf, hf, gt_lf, gt_hf) in enumerate(self.train_loader):
            
            # Move data to device
            lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)
            
            # Clear the gradients from previous batch
            self.optimizer.zero_grad()
            
            # Forward Pass
            loss = self.loss(lf, hf, gt_lf, gt_hf)
            
            # Backpropagate
            loss.backward()
            
            # Update the weights
            self.optimizer.step()
            
            # Accumulate loss for monitoring
            running_loss += loss.item()
            
            #Print progress every 10 batches
            if(batch_idx +1)%10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {running_loss / (batch_idx + 1):.4f}")
            
        return running_loss / len(self.train_loader)
    
    def validate(self) -> float:
        # Switch model to evaluation mode
        self.model.eval()
        val_running_loss = 0.0
        
        if len(self.val_loader) == 0:
            print("No validation data provided. Skipping validation.")
            return 0.0
        
        # No gradient calculation during validation
        with torch.no_grad():
            for batch_idx, (lf, hf, gt_lf, gt_hf) in enumerate(self.val_loader):
                lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)
                
                loss = self.loss(lf, hf, gt_lf, gt_hf)
                val_running_loss += loss.item()
        
        
        return val_running_loss/len(self.val_loader)
    
    def train_model(self):
        print("\n Starting Training")
        last_best = 0
        for epoch in range(self.numEpochs):
            print(f"\nEpoch {epoch + 1}/{self.numEpochs}")
            avg_train_loss = self.train_one_epoch()
            avg_val_loss = self.validate()
            
            print(f"Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                last_best = epoch
                self.save_model()
                print("  [SAVE] New best model saved to vault.")
            
            if (last_best+EARLY_STOPPING)<epoch:
                break
        
        self.load_model()


def train_model(model, trainingParameters, savePoint, emulatedFunction=None):
    optimizer = optim.Adam(model.parameters(), lr=trainingParameters.LEARNING_RATE)

    # Initialize Datasets (Assuming input/GT folder structure) 
    train_dataset = ImageDataset(input_dir=os.path.join(TRAIN_DATA_PATH, "input"), 
                                gt_dir=os.path.join(TRAIN_DATA_PATH, "GT"), 
                                no_gt_dir=os.path.join(TRAIN_DATA_PATH, "nogt"), 
                                emulatedFunction=emulatedFunction) 
    train_loader = DataLoader(train_dataset, batch_size=trainingParameters.BATCH_SIZE, shuffle=True)
    
    val_dataset = ImageDataset(input_dir=os.path.join(VAL_DATA_PATH, "input"), 
                                gt_dir=os.path.join(VAL_DATA_PATH, "GT"),
                                no_gt_dir=os.path.join(TRAIN_DATA_PATH, "nogt"),
                                emulatedFunction=emulatedFunction)
    val_loader = DataLoader(val_dataset, batch_size=trainingParameters.BATCH_SIZE, shuffle=False)

    trainer = UnderwaterTrainer(model, train_loader, val_loader, optimizer, 
                                trainingParameters.NUM_EPOCHS, savePoint)

    trainer.train_model()


if __name__ == '__main__':
    model = ImageEnhancementNetwork()
    model.to(DEVICE)

    # Stage 1: Identity learning with simple L1
    print("Unsupervised Pretraining Phase:")
    train_model(model, UnsupervisedPretrainingParameters, 'checkpoints/identity_model.pth', lambda x: x)
    
    # Stage 2: DCP knowledge transfer with MSE + SSIM
    print("\n\nKnowledge Transfer Phase:")
    train_model(model, KnowledgeTransfer, 'checkpoints/dcp_emulation.pth', enhanceDCP)

    # Stage 3: Full composite loss (MSE + SSIM + Gradient)
    print("\n\nSupervised Training:")
    train_model(model, SupervisedTrainingParameters, 'checkpoints/best_uw_model.pth')