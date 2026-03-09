import torch

from torch.nn import Module, ModuleList, LeakyReLU, Sigmoid, BatchNorm2d, PixelShuffle
from torch.nn import Conv2d

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
        y = self.conv(x)
        if not self.final:
            y = self.bn(y)
        y = self.activation(y)
        return y


class EncoderLayer(Module):
    def __init__(self, in_size, out_size, first=False):
        super(EncoderLayer, self).__init__()
        self.conv = Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        self.bn = BatchNorm2d(out_size)
        self.first = first
        self.activation = LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        y = self.conv(x)
        if not self.first:
            y = self.activation(self.bn(y))
        else:
            y = self.activation(y)
        return y


class DecoderLayer(Module):
    def __init__(self, in_size, out_size, final=False):
        super(DecoderLayer, self).__init__()
        # PixelShuffle requires in_size → out_size * scale^2 channels, then shuffles
        self.conv = Conv2d(in_size, out_size * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle(upscale_factor=2)
        self.bn = BatchNorm2d(out_size)
        self.final = final
        self.activation = Sigmoid() if final else LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        y = self.pixel_shuffle(self.conv(x))
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
        return x + y


class PreliminaryEnhancementNetwork(Module):
    def __init__(self, depth):
        super(PreliminaryEnhancementNetwork, self).__init__()
        self.lfEnhancement = LFEnhancementNetwork(depth)
        self.hfEnhancement = HFEnhancementNetwork(depth)

    def forward(self, lf, hf):
        lf = self.lfEnhancement(lf)
        hf = self.hfEnhancement(hf)
        return lf + hf

class RefinementNetwork(Module):
    def __init__(self, numLayers):
        super(RefinementNetwork, self).__init__()
        self.downscalingLayers = ModuleList(
            EncoderLayer(in_size=3 if i == 0 else 32, out_size=32, first=(i == 0))
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