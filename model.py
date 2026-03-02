import torch

from torch.nn import Module, ModuleList, LeakyReLU, Sigmoid, BatchNorm2d
from torch.nn import Conv2d, ConvTranspose2d

from constants import PRELIMINARY_NETWORK_DEPTH, REFINEMENT_NETWORK_DEPTH
from utilities import reverseTransmissionMap, applyMapBasedAttention


class Layer(Module):
    def __init__(self, in_size, out_size, final=False):
        super(Layer, self).__init__()
        self.conv = Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn = BatchNorm2d(out_size)
        self.final = final

        if final:
            self.activation = Sigmoid()
        else:
            self.activation = LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        y = self.activation(self.bn(self.conv(x)))
        return y


class EncoderLayer(Module):
    def __init__(self, in_size, out_size):
        super(EncoderLayer, self).__init__()
        self.conv = Conv2d(in_size, out_size, kernel_size=2, stride=2)
        self.bn = BatchNorm2d(out_size)
        self.activation = LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class DecoderLayer(Module):
    def __init__(self, in_size, out_size, final=False):
        super(DecoderLayer, self).__init__()
        self.conv = ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.bn = BatchNorm2d(out_size)
        self.final = final
        self.activation = Sigmoid() if final else LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        y = self.conv(x)
        if not self.final:
            y = self.activation(self.bn(y))
        else:
            y = self.activation(y)  # no BN before Sigmoid at output
        return y


class LFEnhancementNetwork(Module):
    def __init__(self, numLayers=4):
        super(LFEnhancementNetwork, self).__init__()
        self.numLayers = numLayers
        self.layers = ModuleList(
            Layer(
                in_size=3 if i == 0 else 32,
                out_size=3 if i == (numLayers - 1) else 32,
                final=i == (numLayers - 1)
            ) for i in range(numLayers)
        )

    def forward(self, x):
        y = x
        for i in range(self.numLayers):
            y = self.layers[i](y)
        return y


class HFEnhancementNetwork(Module):
    def __init__(self, numLayers=4):
        super(HFEnhancementNetwork, self).__init__()
        self.numLayers = numLayers
        self.layers = ModuleList(
            Layer(
                in_size=3 if i == 0 else 32,
                out_size=3 if i == (numLayers - 1) else 32,
                final=i == (numLayers - 1)
            ) for i in range(numLayers)
        )

    def forward(self, x):
        y = x
        for i in range(self.numLayers):
            y = self.layers[i](y)
        # Final layer has Sigmoid so y is in [0,1]; clamp sum to stay in range
        return torch.clamp(x + y, 0, 1)


class PreliminaryEnhancementNetwork(Module):
    def __init__(self, depth):
        super(PreliminaryEnhancementNetwork, self).__init__()
        self.lfEnhancement = LFEnhancementNetwork(depth)
        self.hfEnhancement = HFEnhancementNetwork(depth)

    def forward(self, lf, hf):
        lf = self.lfEnhancement(lf)
        hf = self.hfEnhancement(hf)
        # Average rather than sum to keep output in [0,1]
        return (lf + hf) / 2

class RefinementNetwork(Module):
    def __init__(self, numLayers):
        super(RefinementNetwork, self).__init__()
        self.downscalingLayers = ModuleList(
            EncoderLayer(in_size=3 if i == 0 else 32, out_size=32)
            for i in range(numLayers)
        )
        self.upscalingLayers = ModuleList(
            DecoderLayer(in_size=32, out_size=3 if i == (numLayers - 1) else 32, final=i == (numLayers - 1))
            for i in range(numLayers)
        )
        self.numLayers = numLayers

    def forward(self, x):
        attentionMap = reverseTransmissionMap(x)

        y = x
        for layer_n in range(self.numLayers):
            y = self.downscalingLayers[layer_n](y)
            y = applyMapBasedAttention(y, attentionMap)

        for layer_n in range(self.numLayers):
            if layer_n == self.numLayers - 1:
                y = self.upscalingLayers[layer_n](y)  # final layer — no attention after Sigmoid
            else:
                y = self.upscalingLayers[layer_n](y)
                y = applyMapBasedAttention(y, attentionMap)

        return torch.clamp(y + x, 0, 1)


class ImageEnhancementNetwork(Module):
    def __init__(self, preliminaryNetworkDepth=PRELIMINARY_NETWORK_DEPTH, refinementNetworkDepth=REFINEMENT_NETWORK_DEPTH):
        super(ImageEnhancementNetwork, self).__init__()
        self.preliminaryNetwork = PreliminaryEnhancementNetwork(preliminaryNetworkDepth)
        self.refinementNetwork = RefinementNetwork(refinementNetworkDepth)

    def forward(self, lf, hf):
        preliminary_image = self.preliminaryNetwork(lf, hf)
        enhanced_image = self.refinementNetwork(preliminary_image)
        return enhanced_image


if __name__ == "__main__":
    lf = torch.randn(1, 3, 256, 256)
    hf = torch.randn(1, 3, 256, 256)
    model = ImageEnhancementNetwork()
    y = model(lf, hf)
    print(y.shape)