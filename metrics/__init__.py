from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim
from .uiqm import UIQM

class Evaluator:
    def __init__(self, data: Dataset, model, device):
        self.data = data
        self.loader = DataLoader(data, batch_size=1)
        self.model = model
        self.device = device
        self.metrics = { #Add other metrics here
            "UIQM": UIQM.uiqm,
            "SSIM": lambda x,y: ssim(x, y, data_range=1.0, size_average=True).item(),
            "PSNR": lambda x,y: psnr(x, y, data_range=1.0).item()
        }
    
    def average_metric(self, metricName: str) -> float:
        metric = self.metrics[metricName]
        num = len(self.data)
        assert num>0
        total = 0.0
        num_done = 0
        for lf, hf, gt_lf, gt_hf in self.loader:
            lf, hf, gt_lf, gt_hf = lf.to(self.device), hf.to(self.device), gt_lf.to(self.device), gt_hf.to(self.device)
            total += metric(self.model(lf, hf)[0], gt_lf+gt_hf)
            num_done += 1
            if num_done%20==0:
                print(f"Calculating {metricName}:{num_done}/{num}")

        res =  total/num
        print(f"Result: {metricName}={res}\n")
        return res
    
    def evaluate(self) -> dict:
        result = {}
        for key in self.metrics:
            result[key] = self.average_metric(key)
        return result