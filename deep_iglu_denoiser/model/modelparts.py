import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConvBlock(nn.Module):
    """
    Convolutional block for the U-Net architecture.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Parameters:
        - input: Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.conv_block(input)


class Down(nn.Module):
    """
    Downsample block for the U-Net architecture.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.unet_down_block = nn.Sequential(
            nn.MaxPool2d(2), UnetConvBlock(in_channels, out_channels)
        )

    def forward(self, input) -> torch.Tensor:
        """
        Forward pass through the downsample block.

        Parameters:
        - input: Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.unet_down_block(input)


class Up(nn.Module):
    """
    Upsample block for the U-Net architecture.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = UnetConvBlock(in_channels, out_channels)

    def forward(self, input, input_skip) -> torch.Tensor:
        """
        Forward pass through the upsample block.

        Parameters:
        - input: Input tensor.
        - input_skip: Tensor from the skip connection.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x1 = self.up(input)
        # in case padding is needed
        diff_y = input_skip.size()[2] - x1.size()[2]
        diff_x = input_skip.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )

        x = torch.cat([input_skip, x1], dim=1)
        return self.conv(x)
