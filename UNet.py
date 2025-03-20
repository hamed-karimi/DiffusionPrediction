import torch
import torch.nn as nn


def center_crop_and_concat(up, skip):
    """
    Crops the upsampled tensor to match the size of the skip tensor before concatenation.
    This ensures that the concatenation happens along the channel dimension.
    """
    diff_y = skip.size(2) - up.size(2)
    diff_x = skip.size(3) - up.size(3)

    # Pad upsampled output to match the skip connection size
    up = nn.functional.pad(up, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

    # Concatenate along the channel axis
    return torch.cat([up, skip], dim=1)  # Concatenate along the channel axis


class AdvancedUNet(nn.Module):
    def __init__(self):
        super(AdvancedUNet, self).__init__()

        # Encoder path (downsampling)
        self.enc1 = self.conv_block(9 + 1, 64)     # Input: 3 -> 64
        self.enc2 = self.conv_block(64, 128)   # 64 -> 128
        self.enc3 = self.conv_block(128, 256)  # 128 -> 256
        self.enc4 = self.conv_block(256, 512)  # 256 -> 512
        self.enc5 = self.conv_block(512, 1024) # 512 -> 1024 (optional deep layer)

        # Decoder path (upsampling)
        self.up5 = self.upconv_block(1024, 512)  # 1024 -> 512
        self.up4 = self.upconv_block(1024, 256)   # 512 -> 256
        self.up3 = self.upconv_block(512, 128)   # 256 -> 128
        self.up2 = self.upconv_block(256, 64)    # 128 -> 64
        self.up1 = self.upconv_block(128, 3)      # 64 -> 3 (output channels)
        # self.conv_out = nn.ConvTranspose2d(128, 3, 2, 2)

    def conv_block(self, in_channels, out_channels):
        """Two convolution layers with BatchNorm and LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """Upsampling block with ConvTranspose2d, BatchNorm and LeakyReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, context, timesteps):
        # we might need to normalize timesteps or apply a layer on them
        timesteps = torch.tile(timesteps.view(-1, 1, 1, 1), (1, 1, x.shape[2], x.shape[3]))
        x = torch.cat([x, context, timesteps], dim=1) # concatenate the noisy image and the context of the previous frames
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder path (upsampling) with skip connections
        up5 = self.up5(enc5)
        up5 = center_crop_and_concat(up5, enc4)

        up4 = self.up4(up5)
        up4 = center_crop_and_concat(up4, enc3)

        up3 = self.up3(up4)
        up3 = center_crop_and_concat(up3, enc2)

        up2 = self.up2(up3)
        up2 = center_crop_and_concat(up2, enc1)

        # output = self.conv_out(up2)
        up1 = self.up1(up2)

        # return output
        return up1
