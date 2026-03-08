import torch

from preprocessing import ImageDataset, splitTensor
from model import ImageEnhancementNetwork
from metrics import Evaluator
from constants.TrainingParameters import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluateModel(image):
    lf, hf = splitTensor(lf, hf)
    model = ImageEnhancementNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

if __name__=='__main__':
    test_dataset = ImageDataset(input_dir=TEST_DATA_PATH+'/input',gt_dir=TEST_DATA_PATH+'/GT', limitImages=10)
    eval = Evaluator(test_dataset)

    print(eval.evaluate())