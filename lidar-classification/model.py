import numpy as np
import torch
import torch.functional as F
from torchvision.transforms import CenterCrop
import torchvision.transforms.functional as TF

# Contains 2x Conv & Relu layers [+ an optional Batch Norm layer]
class ConvRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize):
        super(ConvRelu, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.activation = torch.nn.ReLU(inplace=True)

        self.normalize = normalize

    def forward(self, x):
        x = self.activation(self.conv1(x))
        if self.normalize:
            x = self.batch_norm(x)
        x = self.activation(self.conv2(x))
        return x

# Runs a 2x Conv & Relu
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.convRelu = ConvRelu(in_channels, out_channels, normalize=True)

    def forward(self, x):
        x = self.convRelu(x)
        return x

# Runs a Crop + 2x Conv & Relu
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.convRelu = ConvRelu(in_channels, out_channels, normalize=False)

    def forward(self, x1, x2):
        # x = torch.cat((x1, CenterCrop(tuple(x1.shape[-2:]))(x2)), dim=1)
        x = torch.cat((TF.resize(x1, x2.shape[-2:]), x2), dim=1)
        x = self.convRelu(x)
        return x

class UNet(torch.nn.Module):

    def __init__(self, num_classes, in_channels = 1):
        super(UNet, self).__init__()

        self.num_classes = num_classes if num_classes > 2 else 1
        self.channels = [in_channels, 16, 32, 64]

        # Encoding layers (down)
        self.encoder_blocks = [EncoderBlock(self.channels[i], self.channels[i+1])
            for i in range(len(self.channels)-1)]
        self.max_pool = torch.nn.MaxPool2d(2, 2)

        # Decoding layers (up)
        self.up_convs = [torch.nn.ConvTranspose2d(self.channels[i], self.channels[i-1], 2, stride=2)
            for i in range(len(self.channels)-1, 1, -1)]
        self.decoder_blocks = [DecoderBlock(self.channels[i], self.channels[i-1])
            for i in range(len(self.channels)-1, 1, -1)]

        self.output_conv = torch.nn.Conv2d(self.channels[1], self.num_classes, 1)


    def forward(self, x):
        # TODO: Implement as loop based on # blocks in encoder_blocks
        # Encoding phase
        x1 = self.encoder_blocks[0](x)

        x2 = self.max_pool(x1)
        x2 = self.encoder_blocks[1](x2)

        x3 = self.max_pool(x2)
        x3 = self.encoder_blocks[2](x3)

        # Decoding phase
        x = self.up_convs[0](x3)
        x = self.decoder_blocks[0](x, x2)

        x = self.up_convs[1](x)
        x = self.decoder_blocks[1](x, x1)

        x = self.output_conv(x)

        if self.num_classes <= 2:
            return np.squeeze(x)
        else:
            return x
