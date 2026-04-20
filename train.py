import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from seed import set_seed
from utilities import enhanceDCP
from loss import CompositeLoss
from model import ImageEnhancementNetwork
from preprocessing import ImageDataset
from constants.TrainingParameters import *


# Configs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnderwaterTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, numEpochs, modelSavePoint):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = DEVICE
        self.numEpochs = numEpochs
        self.savePoint = modelSavePoint
        self.loss = CompositeLoss(model).to(DEVICE)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=numEpochs, eta_min=1e-6)
        if self.device==torch.device('cuda'):
            self.scaler = torch.amp.GradScaler('cuda')

        self.best_val_loss = float('inf')

    def save_model(self):
        if self.savePoint:
            parent_dir = os.path.dirname(self.savePoint)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
                
            # Strip the 'module.' prefix if using DataParallel
            state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
            torch.save(state_dict, self.savePoint)
    
    def load_model(self):
        if self.savePoint:
            state_dict = torch.load(self.savePoint, map_location=DEVICE, weights_only=False)
            
            # Route the weights correctly whether using DataParallel or not
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
    def train_one_epoch(self) -> float:
        self.model.train()
        self.loss.train()
        running_loss = 0.0
        
        for batch_idx, (lf, hf, gt_lf, gt_hf) in enumerate(self.train_loader):
            lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)
            self.optimizer.zero_grad()
            
            if self.device!=torch.device('cuda'):
                loss = self.loss(lf, hf, gt_lf, gt_hf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            else:
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    loss = self.loss(lf, hf, gt_lf, gt_hf)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            running_loss += loss.item()
            
            if(batch_idx +1)%10 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {running_loss / (batch_idx + 1):.4f}")
            
        return running_loss / len(self.train_loader)
    
    def validate(self) -> float:
        self.model.eval()
        val_running_loss = 0.0
        
        if len(self.val_loader) == 0:
            print("No validation data provided. Skipping validation.")
            return 0.0
        
        with torch.no_grad():
            psnr_sum, ssim_sum, de_sum, n = 0.0, 0.0, 0.0, 0
            for batch_idx, (lf, hf, gt_lf, gt_hf) in enumerate(self.val_loader):
                lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)
                
                loss = self.loss(lf, hf, gt_lf, gt_hf)
                val_running_loss += loss.item()

                gt = (gt_lf + gt_hf).to(self.device)
                out, _, _ = self.model(lf, hf)
                mse = F.mse_loss(out, gt).item()
                psnr_sum += 10 * np.log10(1.0 / (mse + 1e-8))
                ssim_sum += ssim(out, gt, data_range=1.0, size_average=True).item()
                lab_out = KC.rgb_to_lab(out); lab_gt = KC.rgb_to_lab(gt)
                de_sum  += (lab_out - lab_gt).pow(2).sum(dim=1).sqrt().mean().item()
                n += 1

        print(f"  [Metrics] PSNR: {psnr_sum/n:.2f} dB | SSIM: {ssim_sum/n:.4f} | ΔE: {de_sum/n:.4f}")
        return val_running_loss/len(self.val_loader)
    
    def train_model(self):
        print("\n Starting Training")
        last_best = 0
        for epoch in range(self.numEpochs):
            print(f"\nEpoch {epoch + 1}/{self.numEpochs}")
            avg_train_loss = self.train_one_epoch()
            avg_val_loss = self.validate()
            
            print(f"Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            print(f"  [LR] Current learning rate: {new_lr:.2e}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                last_best = epoch
                self.save_model()
                print("  [SAVE] New best model saved to vault.")
            
            if (last_best+EARLY_STOPPING)<=epoch:
                break
        
        self.load_model()

def train_model(model, trainingParameters, savePoint, emulatedFunction=None, limitImages=None):
    optimizer = optim.AdamW(model.parameters(), lr=trainingParameters.LEARNING_RATE, weight_decay=1e-4)

    full_dataset = ImageDataset(input_dir=os.path.join(TRAIN_DATA_PATH, "input"), 
                                gt_dir=os.path.join(TRAIN_DATA_PATH, "GT"), 
                                no_gt_dir=os.path.join(TRAIN_DATA_PATH, "nogt"), 
                                emulatedFunction=emulatedFunction,
                                limitImages=limitImages)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=trainingParameters.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainingParameters.BATCH_SIZE, shuffle=False)

    trainer = UnderwaterTrainer(model, train_loader, val_loader, optimizer, 
                                trainingParameters.NUM_EPOCHS, savePoint)
    trainer.train_model()


if __name__ == '__main__':
    set_seed(42)
    model = ImageEnhancementNetwork()
    model.to(DEVICE)

    # Stage 1: Identity learning with simple L1
    print("Unsupervised Pretraining Phase:")
    train_model(model, UnsupervisedPretrainingParameters, 'checkpoints/identity_model.pth', lambda x: x[:3], limitImages=10)
    
    # Stage 2: DCP knowledge transfer with MSE + SSIM
    print("\n\nKnowledge Transfer Phase:")
    train_model(model, KnowledgeTransfer, 'checkpoints/dcp_emulation.pth', lambda x: enhanceDCP(x[:3]), limitImages=10)

    # Stage 3: Full composite loss (MSE + SSIM + Gradient)
    print("\n\nSupervised Training:")
    train_model(model, SupervisedTrainingParameters, 'checkpoints/best_uw_model.pth', limitImages=10)