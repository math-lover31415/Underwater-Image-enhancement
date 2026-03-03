from torch.utils.data import DataLoader
from preprocessing import ImageDataset
from metrics.uiqm import UIQM
from constants.TrainingParameters import *

if __name__=='__main__':
    test_dataset = ImageDataset(input_dir=TEST_DATA_PATH+'/input',gt_dir=TEST_DATA_PATH+'/GT')
    test_loader = DataLoader(test_dataset,batch_size=1)
    for lf, hf, gt in test_loader:
        image = lf + hf
        print(UIQM(image.squeeze(0)).uiqm())