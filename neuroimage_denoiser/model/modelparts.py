import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input) -> torch.Tensor:
        return self.conv_block(input)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.unet_down_block = nn.Sequential(
            nn.MaxPool3d((0, 2, 2)), UnetConvBlock(in_channels, out_channels)
        )

    def forward(self, input) -> torch.Tensor:
        return self.unet_down_block(input)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.conv = UnetConvBlock(in_channels, out_channels)

    def forward(self, input, input_skip) -> torch.Tensor:
        x1 = self.up(input)

        # Handle padding for all three dimensions
        diff_y = input_skip.size()[3] - x1.size()[3]
        diff_x = input_skip.size()[4] - x1.size()[4]

        diff_y = input_skip.size()[3] - x1.size()[3]
        diff_x = input_skip.size()[4] - x1.size()[4]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(
                x1,
                (
                    0,
                    0,
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ),
            )

        x = torch.cat([input_skip, x1], dim=1)
        return self.conv(x)
