from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim
from .uiqm import UIQM
import time

class Evaluator:
    def __init__(self, data: Dataset, model, device, displayProgress=False):
        self.data = data
        self.loader = DataLoader(data, batch_size=1)
        self.model = model
        self.device = device
        self.displayProgress = displayProgress
        self.metrics = { #Add other metrics here
            "UIQM": lambda x,y: UIQM.uiqm(x).item(),
            "SSIM": lambda x,y: ssim(x, y, data_range=1.0, size_average=True).item(),
            "PSNR": lambda x,y: psnr(x, y, data_range=1.0).item(),
        }
        self.avgTime = 0
        self.numSamples = 0
    
    def average_metric(self, metricName: str) -> float:
        metric = self.metrics[metricName]
        num = len(self.data)
        assert num>0
        total = 0.0
        num_done = 0
        for lf, hf, gt_lf, gt_hf in self.loader:
            lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)

            # Evaluate
            ti = time.time()
            out = self.model(lf, hf)[0]
            tf = time.time()
            time_taken = (tf-ti)*1000
            self.avgTime += (time_taken - self.avgTime) / (self.numSamples + 1)
            self.numSamples += 1

            total += metric(out, gt_lf+gt_hf)
            num_done += 1
            if num_done%20==0:
                if self.displayProgress:
                    print(f"Calculating {metricName}:{num_done}/{num}")

        res =  total/num
        if self.displayProgress:
            print(f"Result: {metricName}={res}\n")
        return res
    
    def evaluate(self) -> dict:
        result = {}
        self.avgTime = 0
        self.numSamples = 0
        result["Parameters: "] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        for key in self.metrics:
            result[key] = self.average_metric(key)
        result["Time Taken: "] = self.avgTime
        return result