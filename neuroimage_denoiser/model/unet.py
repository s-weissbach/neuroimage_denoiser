import neuroimage_denoiser.model.modelparts as modelparts
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    U-Net architecture for image denoising.

    Parameters:
    - n_channels (int): Number of input channels.
    """

    def __init__(self, n_channels: int) -> None:
        """
        Initialize the U-Net model.

        Parameters:
        - n_channels (int): Number of input channels.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.in_block = modelparts.UnetConvBlock(n_channels, 64)
        self.down1 = modelparts.Down(64, 128)
        self.down2 = modelparts.Down(128, 256)
        self.down3 = modelparts.Down(256, 512)
        self.down4 = modelparts.Down(512, 1024)
        self.up1 = modelparts.Up(1024, 512)
        self.up2 = modelparts.Up(512, 256)
        self.up3 = modelparts.Up(256, 128)
        self.up4 = modelparts.Up(128, 64)
        self.out_block = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, input) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Parameters:
        - input: Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x1 = self.in_block(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.out_block(x)
        return output
