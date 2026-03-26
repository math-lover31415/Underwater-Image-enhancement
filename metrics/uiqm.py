import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import kornia.filters as KF
from math import ceil, floor

class UIQM:
    def __assymetric_trimmed_stats(image: torch.Tensor, alpha_low=0.1, alpha_high=0.1) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, w = image.shape
        k = h*w
        img_flat = image.flatten(start_dim=1)
        t_l = ceil(alpha_low*k)
        t_r = floor(alpha_high*k)
        sorted_vals,_ = torch.sort(img_flat, dim=1)
        trimmed = sorted_vals[:, t_l:k-t_r]
        mean = trimmed.mean(dim=1)
        variance = trimmed.var(dim=1)
        return mean,variance

    def uicm(image: torch.Tensor) -> torch.Tensor:
        rg = image[:, 0]-image[:, 1]
        yb = (image[:, 0]+image[:, 1])/2-image[:, 2]
        mean_rg, var_rg = UIQM.__assymetric_trimmed_stats(rg)
        mean_yb, var_yb = UIQM.__assymetric_trimmed_stats(yb)
        mean_term = (mean_rg.pow(2)+mean_yb.pow(2)).sqrt()
        var_term = (var_rg+var_yb).sqrt()
        return -0.0268 * mean_term + 0.1586 * var_term


    def __eme(image: torch.Tensor, k= 8, eps= 1e-5) -> torch.Tensor:
        x = image.unsqueeze(1)
        i_max = F.max_pool2d(x, kernel_size=k, stride=k)
        i_min = -F.max_pool2d(-x, kernel_size=k, stride=k)
        num_blocks = i_max[0].numel()
        ratio = i_max/i_min.clamp(min=eps)
        log_ratio = ratio.log()
        return (2 / num_blocks) * log_ratio.sum(dim=(1, 2, 3))

    def uism(image: torch.Tensor) -> torch.Tensor:
        lambdas = [0.299,0.587,0.114]
        res = 0
        sobel = lambda c: KF.sobel(c.unsqueeze(1)).squeeze(1)
        for i in range(3):
            res += UIQM.__eme(sobel(image[:,i])*image[:, i])*lambdas[i]
        return res

    def __plipAdd(i: torch.Tensor, j: torch.Tensor, gamma=1026.0) -> torch.Tensor:
        return i + j - (i * j) / gamma
    
    def __plipSub(i: torch.Tensor, j: torch.Tensor, k=1026.0) -> torch.Tensor:
        return k * (i - j) / (k - j)
    
    def __plipMul(i: torch.Tensor, j: torch.Tensor, gamma=1026.0) -> torch.Tensor:
        return gamma - gamma * ((1 - j / gamma) ** i)

    def uiconm(image: torch.Tensor, k=8, eps=1e-5) -> torch.Tensor:
        gray = FV.rgb_to_grayscale(image)
        i_max = F.max_pool2d(gray,kernel_size=k,stride=k)
        i_min = -F.max_pool2d(-gray,kernel_size=k,stride=k)
        numerator = UIQM.__plipSub(i_max,i_min)
        denominator = UIQM.__plipAdd(i_max,i_min).clamp(min=eps)
        ratio = numerator/denominator
        ratio = ratio.clamp(eps)
        logamee = ratio*torch.log(ratio)
        return UIQM.__plipMul(1/logamee[0].numel(),logamee.sum(dim=(1,2,3)))

    def uiqm(image: torch.Tensor, c1=0.0282, c2=0.2953, c3=3.5753) -> torch.Tensor:
        return c1*UIQM.uicm(image) + c2*UIQM.uism(image) + c3*UIQM.uiconm(image)

if __name__=='__main__':
    image = torch.rand((2, 3, 256, 256))
    print(UIQM.uiqm(image))