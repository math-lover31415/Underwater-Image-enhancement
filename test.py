import matplotlib.pyplot as plt
from constants.TrainingParameters import *
from preprocessing import *
from model import *
from utilities import enhanceDCP, apply_clahe_rgb
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = ImageDataset(input_dir=TEST_DATA_PATH+'/input',gt_dir=TEST_DATA_PATH+'/GT') 

# Load the model
model = ImageEnhancementNetwork()
model.load_state_dict(torch.load('checkpoints/best_uw_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def test_image(name,val):
    # Get one sample from the training dataset
    sample_lf, sample_hf, sample_gt_lf, sample_gt_hf = val
    sample_gt = sample_gt_hf+sample_gt_lf
    sample_lf = sample_lf.unsqueeze(0).to(DEVICE)
    sample_hf = sample_hf.unsqueeze(0).to(DEVICE)

    # Enhance the image
    with torch.no_grad():
        enhanced = model(sample_lf, sample_hf).cpu().squeeze(0)

    # Convert tensors to numpy arrays for display
    def tensor_to_img(tensor):
        img = tensor.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        img = np.clip(img, 0, 1)
        return img

    plt.figure(figsize=(12,5))
    plt.subplot(1,5,1)
    plt.title("Original Input")
    plt.imshow(tensor_to_img((sample_lf+sample_hf).squeeze(0)))
    plt.axis('off')
    # Save the enhanced image using PIL

    plt.subplot(1,5,2)
    plt.title("Enhanced with DCP")
    dcp_enhanced = enhanceDCP(sample_lf+sample_hf)
    plt.imshow(tensor_to_img(dcp_enhanced.squeeze(0)))
    plt.axis('off')

    plt.subplot(1,5,3)
    plt.title("Enhanced with Clahe")
    img = tensor_to_img((sample_lf+sample_hf).squeeze(0))
    plt.imshow(apply_clahe_rgb(img))
    plt.axis('off')

    plt.subplot(1,5,4)
    plt.title("Enhanced with model")
    plt.imshow(tensor_to_img(enhanced))
    plt.axis('off')

    plt.subplot(1,5,5)
    plt.title("Ground Truth")
    plt.imshow(tensor_to_img(sample_gt))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"data/{name}.png")  # Save the figure to a file instead of showing it interactively
    # plt.show()  # Commented out to avoid the warning
    plt.close()

for idx in range(len(test_dataset)):
    test_image(f"output/{idx}", test_dataset[idx])