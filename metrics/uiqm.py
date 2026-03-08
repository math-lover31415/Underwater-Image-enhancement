import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import kornia.filters as KF
from math import ceil, floor, sqrt

class UIQM:
    def __assymetric_trimmed_stats(image, alpha_low=0.1, alpha_high=0.1):
        h, w = image.shape
        k = h*w
        img_flat = image.flatten()
        t_l = ceil(alpha_low*k)
        t_r = floor(alpha_high*k)
        sorted_vals,_ = torch.sort(img_flat)
        trimmed = sorted_vals[t_l:k-t_r]
        mean = float(trimmed.mean())
        square_diff = (trimmed-mean)**2
        variance = float(square_diff.mean())
        return mean,variance

    def uicm(image, gt=None):
        rg = image[0]-image[1]
        yb = (image[0]+image[1])/2-image[2]
        mean_rg, var_rg = UIQM.__assymetric_trimmed_stats(rg)
        mean_yb, var_yb = UIQM.__assymetric_trimmed_stats(yb)
        mean_term = sqrt(mean_rg**2+mean_yb**2)
        var_term = sqrt(var_rg+var_yb)
        return -0.0268*mean_term+0.1586*var_term


    def __eme(image, k= 8, eps= 1e-5):
        x = image.unsqueeze(0).unsqueeze(0)
        i_max = F.max_pool2d(x, kernel_size=k, stride=k)
        i_min = -F.max_pool2d(-x, kernel_size=k, stride=k)
        num_blocks = i_max.numel()
        ratio = i_max/i_min.clamp(min=eps)
        log_ratio = torch.log(ratio)
        return float((2/num_blocks)*log_ratio.sum())

    def uism(image, gt=None):
        lambdas = [0.299,0.587,0.114]
        res = 0
        sobel = lambda c: KF.sobel(c.unsqueeze(0).unsqueeze(0)).squeeze()
        for i in range(3):
            res += UIQM.__eme(sobel(image[i])*image[i])*lambdas[i]
        return res

    def __plipAdd(i,j,gamma=1026.0):
        return i+j-(i*j)/gamma
    
    def __plipSub(i,j,k=1026.0):
        return k*(i-j)/(k-j)
    
    def __plipMul(i,j,gamma=1026.0):
        return gamma-gamma*((1-j/gamma)**i)

    def uiconm(image, gt=None, k=8, eps=1e-5):
        gray = FV.rgb_to_grayscale(image).unsqueeze(0)
        i_max = F.max_pool2d(gray,kernel_size=k,stride=k)
        i_min = -F.max_pool2d(-gray,kernel_size=k,stride=k)
        numerator = UIQM.__plipSub(i_max,i_min)
        denominator = UIQM.__plipAdd(i_max,i_min).clamp(min=eps)
        ratio = numerator/denominator
        ratio = ratio.clamp(eps)
        logamee = ratio*torch.log(ratio)
        return float(UIQM.__plipMul(1/logamee.numel(),logamee.sum()))

    def uiqm(image, gt=None, c1=0.0282, c2=0.2953, c3=3.5753):
        if image.ndim==4:
            image = image[0]
        return c1*UIQM.uicm(image) + c2*UIQM.uism(image) + c3*UIQM.uiconm(image)

if __name__=='__main__':
    image = torch.rand((3,256,256))
    print(UIQM.uiqm(image))