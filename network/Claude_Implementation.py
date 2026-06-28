import torch
import torch.nn as nn
import torch.nn.functional as F


class FDEUNet(nn.Module):
    """
    FDE U-Net without ACMix module - includes CBAM and HWD
    """

    def __init__(self, num_channels=3, num_classes=1):
        super(FDEUNet, self).__init__()
        self.encoder = SegEncoder(num_channels)
        self.decoder = SegDecoder(num_classes)

    def forward(self, x):
        # Extract multi-scale features from encoder
        features = self.encoder(x)  # Returns [x1, x2, x3, x4, x5]

        # Decode to get segmentation result
        seg_result = self.decoder(features)

        # Resize to match input size if needed
        if seg_result.size()[2:] != x.size()[2:]:
            seg_result = F.interpolate(seg_result, size=x.size()[2:],
                                       mode='bilinear', align_corners=False)

        return seg_result


class SegEncoder(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        # Haar wavelet downsample modules
        self.haarDownsample2x1 = HaarDownsample2x(16)
        self.haarDownsample2x2 = HaarDownsample2x(32)
        self.haarDownsample2x3 = HaarDownsample2x(64)
        self.haarDownsample2x4 = HaarDownsample2x(128)

        # Initial convolution
        self.pre_layer = nn.Sequential(
            ConvBNReLU(num_channels, 16)
        )

        # Encoder layers with residual blocks and CBAM
        self.layer1 = nn.Sequential(
            ConvBNReLU(16, 16),
            ConvBNReLU(16, 16),
            ResidualBlockWith1x1(16, 16),
            CBAM(16)
        )

        self.layer2 = nn.Sequential(
            ConvBNReLU(32, 32),
            ConvBNReLU(32, 32),
            ResidualBlockWith1x1(32, 32),
            CBAM(32)
        )

        self.layer3 = nn.Sequential(
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
            ResidualBlockWith1x1(64, 64),
            CBAM(64)
        )

        self.layer4 = nn.Sequential(
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
            ResidualBlockWith1x1(128, 128),
            CBAM(128)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.pre_layer(x)  # (B,16,H,W)

        # Layer 1: Process and store for skip connection
        x1 = self.layer1(x)  # (B,16,H,W)

        # Haar wavelet downsample and continue
        x = self.haarDownsample2x1(x1)  # (B,32,H/2,W/2)
        x2 = self.layer2(x)  # (B,32,H/2,W/2)

        x = self.haarDownsample2x2(x2)  # (B,64,H/4,W/4)
        x3 = self.layer3(x)  # (B,64,H/4,W/4)

        x = self.haarDownsample2x3(x3)  # (B,128,H/8,W/8)
        x4 = self.layer4(x)  # (B,128,H/8,W/8)

        x5 = self.haarDownsample2x4(x4)  # (B,256,H/16,W/16)

        return [x1, x2, x3, x4, x5]  # Multi-scale features for decoder


class SegDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Residual blocks for each level
        self.ResidualBlockWith1x1_1 = ResidualBlockWith1x1(16, 16)
        self.ResidualBlockWith1x1_2 = ResidualBlockWith1x1(32, 32)
        self.ResidualBlockWith1x1_3 = ResidualBlockWith1x1(64, 64)
        self.ResidualBlockWith1x1_4 = ResidualBlockWith1x1(128, 128)
        self.ResidualBlockWith1x1_5 = ResidualBlockWith1x1(256, 256)

        # Upsampling layers
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64)
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            ConvBNReLU(32, 32),
            ConvBNReLU(32, 32)
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            ConvBNReLU(16, 16),
            ConvBNReLU(16, 16)
        )

        # Final classification layer
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        # Process deepest features
        x = self.ResidualBlockWith1x1_5(x5)  # (B,256,H/16,W/16)

        # Upsample and add skip connections
        x = self.up4(x)  # (B,128,H/8,W/8)
        x = x + x4  # Skip connection
        x = self.conv4(x)
        x = self.ResidualBlockWith1x1_4(x)

        x = self.up3(x)  # (B,64,H/4,W/4)
        x = x + x3  # Skip connection
        x = self.conv3(x)
        x = self.ResidualBlockWith1x1_3(x)

        x = self.up2(x)  # (B,32,H/2,W/2)
        x = x + x2  # Skip connection
        x = self.conv2(x)
        x = self.ResidualBlockWith1x1_2(x)

        x = self.up1(x)  # (B,16,H,W)
        x = x + x1  # Skip connection
        x = self.conv1(x)
        x = self.ResidualBlockWith1x1_1(x)

        # Final prediction
        out = self.final_conv(x)  # (B,num_classes,H,W)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlockWith1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv for channel adjustment if needed
        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.identity_conv = nn.Identity()

    def forward(self, x):
        identity = self.identity_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # Residual connection
        out = self.relu(out)
        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel attention and spatial attention sequentially.
    """

    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        # Channel attention
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        # Spatial attention
        padding = (sa_kernel - 1) // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=sa_kernel,
                                 padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(max_pool))
        x = x * ca

        # Spatial attention
        avg = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True).values
        sa = torch.sigmoid(self.spatial(torch.cat([avg, max_pool], dim=1)))
        x = x * sa
        return x


class HaarDownsample2x(nn.Module):
    """
    Haar wavelet downsampling with channel increase ×2.
    """

    def __init__(self, in_channels: int, normalize: bool = True):
        super().__init__()
        s = (1 / 2 ** 0.5) if normalize else 1.0

        # Haar basis
        low = torch.tensor([1., 1.]) * s
        high = torch.tensor([1., -1.]) * s
        LL = torch.outer(low, low)
        LH = torch.outer(low, high)
        HL = torch.outer(high, low)
        HH = torch.outer(high, high)
        k = torch.stack([LL, LH, HL, HH], dim=0)  # (4,2,2)

        self.register_buffer("kernels", k.view(4, 1, 2, 2))  # (4,1,2,2)

        # 1×1 conv to compress 4C → 2C
        self.reduce = nn.Conv2d(4 * in_channels, 2 * in_channels,
                                kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Expand kernels for all channels
        kernels = self.kernels.expand(-1, c, -1, -1)  # (4,C,2,2)
        kernels = kernels.permute(1, 0, 2, 3).reshape(4 * c, 1, 2, 2)  # (4C,1,2,2)

        # Apply convolution with stride 2
        y = F.conv2d(x, kernels, stride=2, groups=c)  # (B,4C,H/2,W/2)
        return self.reduce(y)  # (B,2C,H/2,W/2)


# Training setup example
class FDEUNetTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_step(self, images, masks):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(images)
        loss = self.criterion(outputs, masks)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_step(self, images, masks):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Convert to predictions
            preds = torch.sigmoid(outputs) > 0.5

            # Calculate IoU
            intersection = (preds & masks.bool()).float().sum((1, 2, 3))
            union = (preds | masks.bool()).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean()

        return loss.item(), iou.item()


# Usage example:
if __name__ == "__main__":
    # Create model
    model = FDEUNet(num_channels=10, num_classes=1)  # 10 channels for Landsat-8

    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 10, 256, 256)  # Batch=2, Channels=10, Height=256, Width=256
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Initialize trainer
    trainer = FDEUNetTrainer(model)
print("Model ready for training!")