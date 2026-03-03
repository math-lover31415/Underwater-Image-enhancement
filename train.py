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
    """Composite loss: MSE + SSIM + Gradient Difference"""
    def __init__(self, lambda_mse=1.0, lambda_ssim=0.5, lambda_gradient=0.3):
        super(CompositeLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
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
            
            grad_loss += self.l1(out_grad_x, gt_grad_x) + self.l1(out_grad_y, gt_grad_y)
        
        return grad_loss / output.shape[1]
    
    def forward(self, output, gt):
        l_mse = self.mse(output, gt)
        l_ssim = self.ssim_loss(output, gt)
        l_gradient = self.gradient_loss(output, gt)
        
        return self.lambda_mse * l_mse + self.lambda_ssim * l_ssim + self.lambda_gradient * l_gradient


class UnderwaterTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, numEpochs, modelSavePoint, criterion=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = DEVICE
        self.numEpochs = numEpochs
        self.savePoint = modelSavePoint
        
        # Use provided criterion or default to CompositeLoss
        self.criterion = criterion if criterion is not None else CompositeLoss().to(DEVICE)
        self.best_val_loss = float('inf')

        
    def save_model(self):
        if self.savePoint:
            torch.save(self.model.state_dict(), self.savePoint)    
    
    def train_one_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (lf, hf, gt) in enumerate(self.train_loader):
            
            # Move data to device
            lf, hf, gt = lf.to(self.device), hf.to(self.device), gt.to(self.device)
            
            # Clear the gradients from previous batch
            self.optimizer.zero_grad()
            
            # Forward Pass
            output = self.model(lf, hf)
            
            # Calculate Loss and backpropagate
            loss = self.criterion(output, gt)
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
            for batch_idx, (lf, hf, gt) in enumerate(self.val_loader):
                lf, hf, gt = lf.to(self.device), hf.to(self.device), gt.to(self.device)
                
                output = self.model(lf, hf)
                
                loss = self.criterion(output, gt)
                val_running_loss += loss.item()
        
        
        return val_running_loss/len(self.val_loader)
    
    def train_model(self):
        print("\n Starting Training")
        for epoch in range(self.numEpochs):
            print(f"\nEpoch {epoch + 1}/{self.numEpochs}")
            avg_train_loss = self.train_one_epoch()
            avg_val_loss = self.validate()
            
            print(f"Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model()
                print("  [SAVE] New best model saved to vault.")


def train_model(model, trainingParameters, savePoint, emulatedFunction=None, criterion=None):
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
                                trainingParameters.NUM_EPOCHS, savePoint, criterion=criterion)

    trainer.train_model()


if __name__ == '__main__':
    model = ImageEnhancementNetwork()
    model.to(DEVICE)

    # Stage 1: Identity learning with simple L1
    print("Unsupervised Pretraining Phase:")
    train_model(model, UnsupervisedPretrainingParameters, 'checkpoints/identity_model.pth',
                lambda x: x, criterion=nn.L1Loss().to(DEVICE))
    
    # Stage 2: DCP knowledge transfer with MSE + SSIM
    print("\n\nKnowledge Transfer Phase:")
    transfer_loss = CompositeLoss(lambda_mse=1.0, lambda_ssim=0.5, lambda_gradient=0.0).to(DEVICE)
    train_model(model, KnowledgeTransfer, 'checkpoints/dcp_emulation.pth', 
                enhanceDCP, criterion=transfer_loss)

    # Stage 3: Full composite loss (MSE + SSIM + Gradient)
    print("\n\nSupervised Training:")
    full_loss = CompositeLoss(lambda_mse=1.0, lambda_ssim=0.5, lambda_gradient=0.3).to(DEVICE)
    train_model(model, SupervisedTrainingParameters, 'checkpoints/best_uw_model.pth',
                criterion=full_loss)