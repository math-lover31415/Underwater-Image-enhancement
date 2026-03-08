from torch.utils.data import Dataset, DataLoader
from .uiqm import UIQM

class Evaluator:
    def __init__(self, data: Dataset):
        self.data = data
        self.loader = DataLoader(data, batch_size=1)
        self.metrics = {
            "UIQM": UIQM.uiqm #Add other metrics here
        }
    
    def average_metric(self, metricName: str) -> float:
        metric = self.metrics[metricName]
        num = 1
        assert num>0
        total = 0.0
        for lf, hf, gt_lf, gt_hf in self.loader:
            total += metric(lf+hf, gt_lf+gt_hf)
        return total/num
    
    def evaluate(self) -> dict:
        result = {}
        for key in self.metrics:
            result[key] = self.average_metric(key)
        return result