import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual

class EDSR(nn.Module):
    def __init__(self, num_blocks=16, channels=64, scale_factor=4):
        super(EDSR, self).__init__()
        self.entry = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.conv_after_res = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor // 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor // 2),
            nn.Conv2d(channels, channels * scale_factor, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor // 2)
        )

        self.exit = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        entry_out = self.entry(x)
        res_out = self.res_blocks(entry_out)
        res_out = self.conv_after_res(res_out)
        res_out += entry_out # Skip connection

        up_out = self.upsample(res_out)

        exit_out = self.exit(up_out)

        return exit_out
