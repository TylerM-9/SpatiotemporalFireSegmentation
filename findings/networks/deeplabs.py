"""
DeepLabV3+ Semantic Segmentation Model

A clean implementation of DeepLabV3+ with ResNet and MobileNetV2 backbones.
Handles batch_size=1 gracefully and includes proper normalization handling.

Usage:
    model = DeepLabV3Plus(num_classes=21, backbone='resnet50', output_stride=16)
    output = model(input_tensor)  # [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from torchvision import models
    from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights
except ImportError:
    raise ImportError("torchvision is required. Install with: pip install torchvision")


# =============================================================================
# Basic Building Blocks
# =============================================================================

class ConvBNReLU(nn.Module):
    """Standard Conv-BN-ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation=dilation, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# =============================================================================
# ASPP Module (Atrous Spatial Pyramid Pooling)
# =============================================================================

class ASPPPooling(nn.Module):
    """Global average pooling branch with proper handling for small batches"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)  # bias=True since no BN
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    Captures multi-scale contextual information through parallel atrous convolutions
    with different dilation rates plus global average pooling.
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        modules = []

        # 1x1 convolution
        modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0))

        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=3,
                                     padding=rate, dilation=rate))

        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# =============================================================================
# Decoder
# =============================================================================

class Decoder(nn.Module):
    """
    DeepLabV3+ Decoder

    Fuses high-level features from ASPP with low-level features from encoder
    using a simple yet effective architecture.
    """
    def __init__(self, num_classes, low_level_channels=256, aspp_channels=256):
        super().__init__()

        # Reduce low-level feature channels
        self.project_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Refine concatenated features
        self.refine = nn.Sequential(
            SeparableConv(aspp_channels + 48, 256, kernel_size=3, padding=1),
            SeparableConv(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low_level_feat, aspp_feat):
        # Project low-level features
        low_level_feat = self.project_low(low_level_feat)

        # Upsample ASPP features to match low-level feature size
        aspp_feat = F.interpolate(aspp_feat, size=low_level_feat.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # Concatenate and refine
        x = torch.cat([aspp_feat, low_level_feat], dim=1)
        x = self.refine(x)
        x = self.classifier(x)

        return x


# =============================================================================
# Backbone Networks
# =============================================================================

class ResNetBackbone(nn.Module):
    """ResNet backbone with controllable output stride"""
    def __init__(self, name='resnet50', output_stride=16, pretrained=True):
        super().__init__()

        if name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.high_level_channels = 2048
        elif name == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            self.high_level_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {name}")

        # Modify stride and dilation for desired output stride
        if output_stride == 8:
            self._modify_resnet_stride(resnet.layer3, stride=1, dilation=2)
            self._modify_resnet_stride(resnet.layer4, stride=1, dilation=4)
        elif output_stride == 16:
            self._modify_resnet_stride(resnet.layer4, stride=1, dilation=2)

        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # Low-level features (256 channels)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # High-level features

        self.low_level_channels = 256

    def _modify_resnet_stride(self, layer, stride, dilation):
        """Modify ResNet layer to use dilation instead of stride"""
        for block in layer:
            # Modify conv2 in bottleneck blocks
            if hasattr(block, 'conv2'):
                block.conv2.stride = (stride, stride)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)

            # Modify downsample if present
            if block.downsample is not None:
                block.downsample[0].stride = (stride, stride)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder
        low_level_feat = self.layer1(x)  # 1/4 resolution
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)  # 1/output_stride resolution

        return low_level_feat, high_level_feat


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone with controllable output stride"""
    def __init__(self, output_stride=16, pretrained=True):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        features = mobilenet.features

        # Modify stride and dilation for desired output stride
        if output_stride == 8:
            features[14].conv[1].stride = (1, 1)
            features[14].conv[1].dilation = (2, 2)
            features[14].conv[1].padding = (2, 2)

            features[15].conv[1].stride = (1, 1)
            features[15].conv[1].dilation = (2, 2)
            features[15].conv[1].padding = (2, 2)

            features[16].conv[1].stride = (1, 1)
            features[16].conv[1].dilation = (4, 4)
            features[16].conv[1].padding = (4, 4)

            features[17].conv[1].stride = (1, 1)
            features[17].conv[1].dilation = (4, 4)
            features[17].conv[1].padding = (4, 4)
        elif output_stride == 16:
            features[14].conv[1].stride = (1, 1)
            features[14].conv[1].dilation = (2, 2)
            features[14].conv[1].padding = (2, 2)

            features[15].conv[1].stride = (1, 1)
            features[15].conv[1].dilation = (2, 2)
            features[15].conv[1].padding = (2, 2)

            features[16].conv[1].stride = (1, 1)
            features[16].conv[1].dilation = (2, 2)
            features[16].conv[1].padding = (2, 2)

            features[17].conv[1].stride = (1, 1)
            features[17].conv[1].dilation = (2, 2)
            features[17].conv[1].padding = (2, 2)

        self.features = features
        self.low_level_channels = 24
        self.high_level_channels = 1280  # Fixed: MobileNetV2 outputs 1280 channels

    def forward(self, x):
        # Extract low-level features (after inverted residual block 3)
        low_level_feat = self.features[:4](x)  # 1/4 resolution, 24 channels

        # Extract high-level features
        high_level_feat = self.features[4:](low_level_feat)  # 1/output_stride, 1280 channels

        return low_level_feat, high_level_feat


# =============================================================================
# DeepLabV3+ Model
# =============================================================================

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Semantic Segmentation Model

    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone network ('resnet50', 'resnet101', 'mobilenet_v2')
        output_stride: Output stride (8 or 16)
        pretrained_backbone: Use ImageNet pretrained weights
        aspp_rates: Atrous rates for ASPP module
            - For 512x512 images: (6, 12, 18) - default
            - For 256x256 images: (3, 6, 9) - recommended
            - For 1024x1024 images: (12, 24, 36) - recommended
        input_size: Expected input size (H, W) - used to auto-adjust ASPP rates if not specified

    Input:
        x: [B, 3, H, W] RGB image tensor

    Output:
        logits: [B, num_classes, H, W] segmentation logits
    """
    def __init__(self,
                 num_classes,
                 backbone='resnet50',
                 output_stride=16,
                 pretrained_backbone=True,
                 aspp_rates=None,
                 input_size=None):
        super().__init__()

        # Auto-adjust ASPP rates based on input size
        if aspp_rates is None:
            if input_size is not None:
                h, w = input_size if isinstance(input_size, tuple) else (input_size, input_size)
                if h <= 256 or w <= 256:
                    aspp_rates = (3, 6, 9)  # Smaller rates for 256x256
                elif h >= 1024 or w >= 1024:
                    aspp_rates = (12, 24, 36)  # Larger rates for 1024x1024
                else:
                    aspp_rates = (6, 12, 18)  # Default for 512x512
            else:
                aspp_rates = (6, 12, 18)  # Default

        print(f"Using ASPP rates: {aspp_rates}")
        super().__init__()

        # Select backbone
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetBackbone(
                name=backbone,
                output_stride=output_stride,
                pretrained=pretrained_backbone
            )
        elif backbone == 'mobilenet_v2':
            self.backbone = MobileNetV2Backbone(
                output_stride=output_stride,
                pretrained=pretrained_backbone
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ASPP module
        self.aspp = ASPP(
            in_channels=self.backbone.high_level_channels,
            out_channels=256,
            atrous_rates=aspp_rates
        )

        # Decoder
        self.decoder = Decoder(
            num_classes=num_classes,
            low_level_channels=self.backbone.low_level_channels,
            aspp_channels=256
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Extract features
        low_level_feat, high_level_feat = self.backbone(x)

        # Apply ASPP
        aspp_feat = self.aspp(high_level_feat)

        # Decode
        output = self.decoder(low_level_feat, aspp_feat)

        # Upsample to input size
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output

    def freeze_bn(self):
        """Freeze BatchNorm layers (useful for fine-tuning with small batches)"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_backbone_params(self):
        """Get backbone parameters for differential learning rates"""
        return self.backbone.parameters()

    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates"""
        return list(self.aspp.parameters()) + list(self.decoder.parameters())


# =============================================================================
# Pretrained Model Loading
# =============================================================================

def load_pretrained_deeplabv3plus(num_classes=21, backbone='resnet50',
                                  weights='COCO_WITH_VOC_LABELS_V1', input_size=None):
    """
    Load fully pretrained DeepLabV3 model from torchvision and convert to DeepLabV3+

    Args:
        num_classes: Number of classes (21 for PASCAL VOC, 19 for Cityscapes, etc.)
        backbone: 'resnet50' or 'resnet101'
        weights: Pretrained weights to load
            - 'COCO_WITH_VOC_LABELS_V1': COCO + VOC trained (21 classes)
            - None: Random initialization with pretrained backbone only
        input_size: Expected input size (H, W) or int - auto-adjusts ASPP rates
            - 256 or (256, 256): Uses rates (3, 6, 9)
            - 512 or (512, 512): Uses rates (6, 12, 18)
            - 1024 or (1024, 1024): Uses rates (12, 24, 36)

    Returns:
        model: DeepLabV3Plus model with pretrained weights
    """
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights

    # Load pretrained DeepLabV3 from torchvision
    if weights == 'COCO_WITH_VOC_LABELS_V1':
        if backbone == 'resnet50':
            pretrained = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        elif backbone == 'resnet101':
            pretrained = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        print(f"Loaded pretrained DeepLabV3 with {backbone} backbone (COCO + VOC trained)")
    else:
        # Just use pretrained backbone
        print(f"Creating model with pretrained {backbone} backbone only")
        return DeepLabV3Plus(num_classes=num_classes, backbone=backbone,
                           output_stride=16, pretrained_backbone=True, input_size=input_size)

    # Create our DeepLabV3+ model with appropriate ASPP rates
    model = DeepLabV3Plus(num_classes=num_classes, backbone=backbone,
                         output_stride=16, pretrained_backbone=False, input_size=input_size)

    # Transfer weights from pretrained DeepLabV3
    # Note: DeepLabV3 and DeepLabV3+ share the backbone and ASPP
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()

    # Filter and transfer compatible weights
    transferred = 0
    for name, param in model_dict.items():
        # Map our model's parameter names to torchvision's
        torch_name = name.replace('backbone.', 'backbone.').replace('aspp.', 'classifier.0.')

        if torch_name in pretrained_dict:
            if param.shape == pretrained_dict[torch_name].shape:
                model_dict[name] = pretrained_dict[torch_name]
                transferred += 1

    model.load_state_dict(model_dict, strict=False)
    print(f"Transferred {transferred} pretrained parameters")

    return model


def adapt_num_classes(model, new_num_classes):
    """
    Adapt a pretrained model to a different number of classes
    Useful for transfer learning (e.g., VOC 21 classes -> your dataset with N classes)

    Args:
        model: Pretrained DeepLabV3Plus model
        new_num_classes: New number of output classes

    Returns:
        model: Model with new classifier head
    """
    # Replace final classifier
    model.decoder.classifier = nn.Conv2d(256, new_num_classes, 1)

    # Initialize new classifier
    nn.init.kaiming_normal_(model.decoder.classifier.weight, mode='fan_out', nonlinearity='relu')
    if model.decoder.classifier.bias is not None:
        nn.init.constant_(model.decoder.classifier.bias, 0)

    print(f"Adapted model to {new_num_classes} classes")
    return model


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    # Test with different configurations
    print("Testing DeepLabV3+ implementations...\n")

    # Test 1: Load pretrained model for 256x256 images
    print("=" * 60)
    print("Loading pretrained model for 256x256 input...")
    print("=" * 60)
    model = load_pretrained_deeplabv3plus(num_classes=21, backbone='resnet50', input_size=256)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Pretrained 256x256 - Input: {x.shape}, Output: {output.shape}\n")

    # Test 2: Adapt pretrained model to custom number of classes (256x256)
    print("=" * 60)
    print("Adapting to custom dataset (5 classes) with 256x256 input...")
    print("=" * 60)
    model = load_pretrained_deeplabv3plus(num_classes=21, backbone='resnet50', input_size=256)
    model = adapt_num_classes(model, new_num_classes=5)
    output = model(x)
    print(f"Adapted 256x256 - Input: {x.shape}, Output: {output.shape}\n")

    # Test 3: Different input sizes with auto ASPP rate adjustment
    print("=" * 60)
    print("Testing different input sizes...")
    print("=" * 60)
    for size in [256, 512, 1024]:
        model = DeepLabV3Plus(num_classes=21, backbone='resnet50',
                             pretrained_backbone=False, input_size=size)
        x_test = torch.randn(1, 3, size, size)
        output = model(x_test)
        print(f"Size {size}x{size} - Input: {x_test.shape}, Output: {output.shape}")
    print()

    # Test 4: MobileNetV2 for 256x256 (faster inference)
    print("=" * 60)
    print("MobileNetV2 for 256x256 (lightweight)...")
    print("=" * 60)
    model = DeepLabV3Plus(num_classes=21, backbone='mobilenet_v2',
                         output_stride=16, pretrained_backbone=True, input_size=256)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"MobileNetV2 256x256 - Input: {x.shape}, Output: {output.shape}\n")

    # Test 5: Batch size 1 with 256x256 (common in inference)
    print("=" * 60)
    print("Testing batch size 1 with 256x256...")
    print("=" * 60)
    x_single = torch.randn(1, 3, 256, 256)
    model.eval()  # Important for batch_size=1
    output = model(x_single)
    print(f"Batch=1, 256x256 - Input: {x_single.shape}, Output: {output.shape}\n")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"Model Statistics")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Show recommended usage
    print("\n" + "=" * 60)
    print("RECOMMENDED USAGE FOR 256x256 IMAGES:")
    print("=" * 60)
    print("# Load pretrained model")
    print("model = load_pretrained_deeplabv3plus(")
    print("    num_classes=21,")
    print("    backbone='resnet50',")
    print("    input_size=256  # Auto-adjusts ASPP rates to (3, 6, 9)")
    print(")")
    print("\n# For custom classes (transfer learning)")
    print("model = adapt_num_classes(model, new_num_classes=YOUR_CLASSES)")
    print("\n# Input should be normalized with ImageNet stats:")
    print("# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")