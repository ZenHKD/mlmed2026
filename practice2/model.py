import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.shortcut(x) + self.double_conv(x)


class Down1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.down(x)


class Up1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip connection: out_channels + out_channels = 2*out_channels
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder
            x2: Skip connection from encoder
        """
        x1 = self.upsample(x1)

        dY = x2.shape[2] - x1.shape[2]
        dX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [dX // 2, dX - dX // 2,
                        dY // 2, dY - dY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Down2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels * 2, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from encoder 
            x2: Skip connection from decoder
        """

        dY = x2.shape[2] - x1.shape[2]
        dX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [dX // 2, dX - dX // 2,
                        dY // 2, dY - dY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return self.pool(x)


class Up2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip connection: out_channels + out_channels = 2*out_channels
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder
            x2: Skip connection from encoder
        """
        x1 = self.upsample(x1)

        dY = x2.shape[2] - x1.shape[2]
        dX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [dX // 2, dX - dX // 2,
                        dY // 2, dY - dY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CU_Net(nn.Module):

    def __init__(
        self, 
        in_channels=1, 
        num_classes=1
        ):
        super().__init__()

        BASE_CHANNELS = 16

        # U-Net 1
        self.enc1_1 = DoubleConv(in_channels, BASE_CHANNELS)
        self.enc1_2 = Down1(BASE_CHANNELS, BASE_CHANNELS*2)
        self.enc1_3 = Down1(BASE_CHANNELS*2, BASE_CHANNELS*4)
        self.enc1_4 = Down1(BASE_CHANNELS*4, BASE_CHANNELS*8)

        self.bottleneck1 = Down1(BASE_CHANNELS*8, BASE_CHANNELS*16)
        
        self.dec1_4 = Up1(BASE_CHANNELS*16, BASE_CHANNELS*8)
        self.dec1_3 = Up1(BASE_CHANNELS*8, BASE_CHANNELS*4)
        self.dec1_2 = Up1(BASE_CHANNELS*4, BASE_CHANNELS*2)
        self.dec1_1 = Up1(BASE_CHANNELS*2, BASE_CHANNELS)

        self.out1_1 = OutConv(BASE_CHANNELS, num_classes)
        self.out1_2 = OutConv(BASE_CHANNELS*2, num_classes)
        self.out1_3 = OutConv(BASE_CHANNELS*4, num_classes)
        self.out1_4 = OutConv(BASE_CHANNELS*8, num_classes)
        self.out1_5 = OutConv(BASE_CHANNELS*16, num_classes)

        # U-Net 2
        self.enc2_1 = DoubleConv(num_classes, BASE_CHANNELS)
        self.enc2_2 = Down2(BASE_CHANNELS, BASE_CHANNELS*2)
        self.enc2_3 = Down2(BASE_CHANNELS*2, BASE_CHANNELS*4)
        self.enc2_4 = Down2(BASE_CHANNELS*4, BASE_CHANNELS*8)

        self.bottleneck2 = Down2(BASE_CHANNELS*8, BASE_CHANNELS*16)
        
        self.dec2_4 = Up2(BASE_CHANNELS*16, BASE_CHANNELS*8)
        self.dec2_3 = Up2(BASE_CHANNELS*8, BASE_CHANNELS*4)
        self.dec2_2 = Up2(BASE_CHANNELS*4, BASE_CHANNELS*2)
        self.dec2_1 = Up2(BASE_CHANNELS*2, BASE_CHANNELS)

        self.out2_1 = OutConv(BASE_CHANNELS, num_classes)
        self.out2_2 = OutConv(BASE_CHANNELS*2, num_classes)
        self.out2_3 = OutConv(BASE_CHANNELS*4, num_classes)
        self.out2_4 = OutConv(BASE_CHANNELS*8, num_classes)
        self.out2_5 = OutConv(BASE_CHANNELS*16, num_classes)

    def forward(self, x):

        # x: [batch_size, channels, height, width]
        target_size = x.shape[2:]
        
        # U-Net 1
        e1_1 = self.enc1_1(x)
        e1_2 = self.enc1_2(e1_1)
        e1_3 = self.enc1_3(e1_2)
        e1_4 = self.enc1_4(e1_3)

        b1 = self.bottleneck1(e1_4)

        d1_4 = self.dec1_4(b1, e1_4)
        d1_3 = self.dec1_3(d1_4, e1_3)
        d1_2 = self.dec1_2(d1_3, e1_2)
        d1_1 = self.dec1_1(d1_2, e1_1)

        o1_1 = self.out1_1(d1_1)
        o1_2 = self.out1_2(d1_2)
        o1_3 = self.out1_3(d1_3)
        o1_4 = self.out1_4(d1_4)
        o1_5 = self.out1_5(b1)

        o1_2 = F.interpolate(o1_2, size=target_size, mode='bilinear', align_corners=True)
        o1_3 = F.interpolate(o1_3, size=target_size, mode='bilinear', align_corners=True)
        o1_4 = F.interpolate(o1_4, size=target_size, mode='bilinear', align_corners=True)
        o1_5 = F.interpolate(o1_5, size=target_size, mode='bilinear', align_corners=True)

        branch1 = o1_1 + o1_2 + o1_3 + o1_4 + o1_5

        # U-Net 2
        e2_1 = self.enc2_1(branch1)
        e2_2 = self.enc2_2(e2_1, d1_1)
        e2_3 = self.enc2_3(e2_2, d1_2)
        e2_4 = self.enc2_4(e2_3, d1_3)

        b2 = self.bottleneck2(e2_4, d1_4)

        d2_4 = self.dec2_4(b2, e2_4)
        d2_3 = self.dec2_3(d2_4, e2_3)
        d2_2 = self.dec2_2(d2_3, e2_2)
        d2_1 = self.dec2_1(d2_2, e2_1)

        o2_1 = self.out2_1(d2_1)
        o2_2 = self.out2_2(d2_2)
        o2_3 = self.out2_3(d2_3)
        o2_4 = self.out2_4(d2_4)
        o2_5 = self.out2_5(b2)

        o2_2 = F.interpolate(o2_2, size=target_size, mode='bilinear', align_corners=True)
        o2_3 = F.interpolate(o2_3, size=target_size, mode='bilinear', align_corners=True)
        o2_4 = F.interpolate(o2_4, size=target_size, mode='bilinear', align_corners=True)
        o2_5 = F.interpolate(o2_5, size=target_size, mode='bilinear', align_corners=True)

        branch2 = o2_1 + o2_2 + o2_3 + o2_4 + o2_5

        # Fusion
        branch = branch1 + branch2

        return branch


if __name__ == "__main__":
    model = CU_Net()
    
    # Dummy input
    x = torch.randn(1, 1, 800, 540)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")