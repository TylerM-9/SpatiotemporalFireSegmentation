"""
Spatio-Temporal DeepLabV3+ Architecture
Integrates temporal coherence with DeepLabV3+ segmentation using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

try:
    from torchvision import models
    from torchvision.models import ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights
except ImportError:
    raise ImportError("torchvision is required. Install with: pip install torchvision")


# =============================================================================
# Basic Building Blocks (from DeepLabV3+)
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
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
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
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()

        modules = []
        modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0))

        for rate in atrous_rates:
            modules.append(ConvBNReLU(in_channels, out_channels, kernel_size=3,
                                     padding=rate, dilation=rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

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
# ATTENTION MODULE (from STCNN)
# =============================================================================

class SimpleContextAdd(nn.Module):
    """
    Attention module that integrates:
      - x:                current stage features              [B, Cx, Hx, Wx]
      - prev:             previous stage features             [B, Cp, Hp, Wp]
      - temporal:         temporal branch features            [B, Ct, Ht, Wt]
      - context_high:     high-level context features         [B, Cc, Hc, Wc]

    Operations:
      1. out_down = x + P(context_high) where P projects context to match x channels
      2. out_forward1 = concat(out_down, P_temp(temporal))
      3. middle_conv = Conv3x3(out_forward1)
      4. out_forward2 = middle_conv * prev (element-wise multiplication)
      5. out = concat(out_forward1, out_forward2)
      6. out = Conv3x3(out)

    Returns:
      - out_down: Direct sum for skip connection
      - out: Attended features after full processing
    """
    def __init__(self, in_channels: int, context_channels: int, temporal_channels: int = None):
        super().__init__()

        if temporal_channels is None:
            temporal_channels = context_channels

        # Projection layer to match context_high channels to x channels
        if in_channels != context_channels:
            self.context_projection = nn.Conv2d(
                context_channels,
                in_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.context_projection = None

        # Projection layer to match temporal channels to x channels
        if in_channels != temporal_channels:
            self.temporal_projection = nn.Conv2d(
                temporal_channels,
                in_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.temporal_projection = None

        # First 3x3 conv: processes concatenated features
        self.conv3x3 = nn.Conv2d(
            in_channels * 2,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        # Second 3x3 conv: final processing
        self.conv3x3_2 = nn.Conv2d(
            in_channels * 3,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, prev, temporal, context_high):
        """
        Args:
            x: Current decoder features [B, C, H, W]
            prev: Previous stage features [B, C, H, W]
            temporal: Temporal branch features [B, C_t, H, W]
            context_high: High-level context [B, C_c, H, W]

        Returns:
            out_down: Simple addition output
            out: Fully attended output
        """
        # Project context_high to match x channels if needed
        if self.context_projection is not None:
            context_high = self.context_projection(context_high)

        # Resize context_high to match x spatial dimensions if needed
        if context_high.shape[2:] != x.shape[2:]:
            context_high = F.interpolate(
                context_high,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 1: Element-wise addition
        out_down = x + context_high

        # Project temporal features to match x channels if needed
        if self.temporal_projection is not None:
            temporal = self.temporal_projection(temporal)

        # Resize temporal to match x spatial dimensions if needed
        if temporal.shape[2:] != x.shape[2:]:
            temporal = F.interpolate(
                temporal,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 2: Concatenate with temporal features
        out_forward1 = torch.cat((out_down, temporal), dim=1)

        # Step 3: First convolution
        middle_conv = self.conv3x3(out_forward1)

        # Step 4: Element-wise multiplication with previous features
        # Ensure prev has same spatial size
        if prev.shape[2:] != middle_conv.shape[2:]:
            prev = F.interpolate(
                prev,
                size=middle_conv.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        out_forward2 = middle_conv * prev

        # Step 5: Concatenate both branches
        out = torch.cat((out_forward1, out_forward2), dim=1)

        # Step 6: Final convolution
        out = self.conv3x3_2(out)

        return out_down, out


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
            if hasattr(block, 'conv2'):
                block.conv2.stride = (stride, stride)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)

            if block.downsample is not None:
                block.downsample[0].stride = (stride, stride)

    def forward(self, x, return_all_features=False):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder
        low_level_feat = self.layer1(x)  # 1/4 resolution, 256 channels
        x2 = self.layer2(low_level_feat)
        x3 = self.layer3(x2)
        high_level_feat = self.layer4(x3)  # 1/output_stride resolution

        if return_all_features:
            # Return all intermediate features for temporal integration
            return low_level_feat, x2, x3, high_level_feat
        else:
            return low_level_feat, high_level_feat


# =============================================================================
# DeepLabV3+ Decoder with Attention
# =============================================================================

class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ Decoder with temporal attention integration.

    The decoder operates in two stages:
    1. ASPP features are upsampled 4x (from 1/16 to 1/4 resolution)
    2. Concatenated with low-level features and refined to full resolution

    Attention is integrated at the concatenation stage (1/4 resolution).
    """
    def __init__(self, num_classes, low_level_channels=256, aspp_channels=256,
                 use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        # Reduce low-level feature channels (standard DeepLabV3+)
        self.project_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Attention module for temporal integration (operates on ASPP features)
        if self.use_attention:
            self.attention = SimpleContextAdd(
                in_channels=aspp_channels,           # 256 (ASPP output)
                context_channels=aspp_channels,       # 256 (from temporal ASPP)
                temporal_channels=aspp_channels       # 256 (from temporal ASPP)
            )

        # Refine concatenated features
        # Input channels: aspp_channels (256) + 48 (projected low-level)
        self.refine = nn.Sequential(
            SeparableConv(aspp_channels + 48, 256, kernel_size=3, padding=1),
            SeparableConv(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

        print(f"DeepLabV3+ Decoder initialized")
        print(f"Attention: {'Enabled (with GroupNorm)' if use_attention else 'Disabled'}")

    def forward(self, low_level_feat, aspp_feat, temporal_aspp=None, prev_aspp=None):
        """
        Forward pass through decoder with optional temporal integration.

        Args:
            low_level_feat: Low-level features from encoder [B, 256, H/4, W/4]
            aspp_feat: ASPP output features [B, 256, H/16, W/16]
            temporal_aspp: Temporal ASPP features [B, 256, H/16, W/16] (optional)
            prev_aspp: Previous frame ASPP features [B, 256, H/16, W/16] (optional)

        Returns:
            output: Segmentation logits [B, num_classes, H/4, W/4] (needs 4x upsample)
            attention_output: Attended features (for auxiliary loss)
        """
        attention_output = None

        # Apply attention if enabled and temporal features provided
        if self.use_attention and temporal_aspp is not None:
            # Use aspp_feat itself as prev if not provided (first frame case)
            if prev_aspp is None:
                prev_aspp = aspp_feat.detach()

            # Apply attention module
            _, aspp_feat = self.attention(
                x=aspp_feat,
                prev=prev_aspp,
                temporal=temporal_aspp,
                context_high=temporal_aspp  # Using temporal as context
            )
            attention_output = aspp_feat

        # Project low-level features
        low_level_feat = self.project_low(low_level_feat)

        # Upsample ASPP features to match low-level feature size (4x upsampling)
        aspp_feat = F.interpolate(aspp_feat, size=low_level_feat.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # Concatenate and refine
        x = torch.cat([aspp_feat, low_level_feat], dim=1)
        x = self.refine(x)
        x = self.classifier(x)

        return x, attention_output


# =============================================================================
# Complete DeepLabV3+ Model
# =============================================================================

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Semantic Segmentation Model with optional temporal attention.

    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone network ('resnet50', 'resnet101')
        output_stride: Output stride (8 or 16)
        pretrained_backbone: Use ImageNet pretrained weights
        aspp_rates: Atrous rates for ASPP module
        input_size: Expected input size (H, W) - auto-adjusts ASPP rates
        use_attention: Enable temporal attention (for spatio-temporal version)
    """
    def __init__(self,
                 num_classes,
                 backbone='resnet50',
                 output_stride=16,
                 pretrained_backbone=True,
                 aspp_rates=None,
                 input_size=None,
                 use_attention=False):
        super().__init__()

        # Auto-adjust ASPP rates based on input size
        if aspp_rates is None:
            if input_size is not None:
                h, w = input_size if isinstance(input_size, tuple) else (input_size, input_size)
                if h <= 256 or w <= 256:
                    aspp_rates = (3, 6, 9)
                elif h >= 1024 or w >= 1024:
                    aspp_rates = (12, 24, 36)
                else:
                    aspp_rates = (6, 12, 18)
            else:
                aspp_rates = (6, 12, 18)

        print(f"Using ASPP rates: {aspp_rates}")

        # Select backbone (ResNet only for now)
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetBackbone(
                name=backbone,
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
        self.decoder = DeepLabV3PlusDecoder(
            num_classes=num_classes,
            low_level_channels=self.backbone.low_level_channels,
            aspp_channels=256,
            use_attention=use_attention
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

    def forward(self, x, temporal_aspp=None, prev_aspp=None):
        """
        Forward pass with optional temporal features.

        Args:
            x: Input tensor [B, 3, H, W]
            temporal_aspp: Temporal ASPP features (optional)
            prev_aspp: Previous frame ASPP features (optional)

        Returns:
            output: Segmentation logits [B, num_classes, H, W]
            attention_output: Attended features (if attention enabled)
        """
        input_size = x.shape[-2:]

        # Extract features
        low_level_feat, high_level_feat = self.backbone(x)

        # Apply ASPP
        aspp_feat = self.aspp(high_level_feat)

        # Decode with optional temporal attention
        output, attention_output = self.decoder(
            low_level_feat, aspp_feat,
            temporal_aspp=temporal_aspp,
            prev_aspp=prev_aspp
        )

        # Upsample to input size (4x from decoder output)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output, attention_output

    def get_aspp_features(self, x):
        """
        Extract ASPP features (useful for temporal branch).

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            aspp_feat: ASPP output features [B, 256, H/16, W/16]
        """
        low_level_feat, high_level_feat = self.backbone(x)
        aspp_feat = self.aspp(high_level_feat)
        return aspp_feat

    def freeze_bn(self):
        """Freeze BatchNorm layers (useful for fine-tuning with small batches)"""
        # Only freeze BN in ASPP and decoder, NOT in backbone
        # Backbone BN must stay in train mode for proper forward pass
        for m in self.aspp.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        print("Froze BatchNorm in ASPP and decoder (backbone BN remains trainable)")

    def get_backbone_params(self):
        """Get backbone parameters for differential learning rates"""
        return self.backbone.parameters()

    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates"""
        return list(self.aspp.parameters()) + list(self.decoder.parameters())


# =============================================================================
# Spatio-Temporal DeepLabV3+ (ST-DeepLabV3+)
# =============================================================================

class STDeepLabV3Plus(nn.Module):
    """
    Spatio-Temporal DeepLabV3+ Network
    Integrates temporal prediction branch with spatial segmentation branch.

    Architecture:
    - pred_encoder: Temporal coherence encoder (processes frame sequence t-4 to t-1)
    - pred_decoder: Temporal coherence decoder (generates predicted frame t and ASPP features)
    - seg_model: Spatial segmentation DeepLabV3+ (uses temporal ASPP features via attention)
    """

    def __init__(self, pred_enc, pred_dec, num_classes=1, backbone='resnet50',
                 output_stride=16, input_size=256):
        """
        Args:
            pred_enc: Temporal prediction encoder (with pretrained weights)
            pred_dec: Temporal prediction decoder (with pretrained weights)
            num_classes: Number of segmentation classes
            backbone: Backbone architecture ('resnet50', 'resnet101')
            output_stride: Output stride (8 or 16)
            input_size: Input image size (for ASPP rate adjustment)
        """
        super().__init__()

        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec

        # Create spatial segmentation DeepLabV3+ with attention enabled
        self.seg_model = DeepLabV3Plus(
            num_classes=num_classes,
            backbone=backbone,
            output_stride=output_stride,
            pretrained_backbone=True,
            input_size=input_size,
            use_attention=True  # Enable temporal attention
        )

        print("ST-DeepLabV3+ initialized with temporal and spatial branches")
        print(f"Backbone: {backbone}, Input size: {input_size}")

    def forward(self, seq, frame):
        """
        Forward pass through complete spatio-temporal network.

        Args:
            seq: Sequence of previous frames [B, T*C, H, W] where T is temporal window
                 e.g., for 4 frames: [B, 12, H, W] if RGB
            frame: Current frame [B, C, H, W]

        Returns:
            seg_output: Segmentation result [B, num_classes, H, W]
            pred: Predicted frame from temporal branch [B, C, H, W]
            attention_output: Attention features (for auxiliary loss)
        """
        # === TEMPORAL BRANCH (TOP) ===
        # Extract temporal features from sequence (t-4, t-3, t-2, t-1)
        pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)

        # Decode temporal features to get prediction
        pred, pred_de_feats = self.pred_decoder(pred_en_feats, return_feature_maps=True)

        # Get temporal ASPP features from predicted frame
        # Detach to prevent gradients flowing back to temporal branch
        with torch.no_grad():
            temporal_aspp = self.seg_model.get_aspp_features(pred.detach())

        # === SPATIAL BRANCH (BOTTOM) ===
        # Process current frame with temporal attention
        seg_output, attention_output = self.seg_model(
            frame,
            temporal_aspp=temporal_aspp,
            prev_aspp=None  # Could add previous frame ASPP if needed
        )

        return seg_output, pred, attention_output

    def load_temporal_weights(self, checkpoint_path):
        """
        Load pretrained weights for temporal branch (encoder + decoder).

        Args:
            checkpoint_path: Path to temporal branch checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load encoder weights
        pred_enc_dict = {k.replace('pred_encoder.', ''): v
                         for k, v in state_dict.items() if 'pred_encoder' in k}
        if pred_enc_dict:
            self.pred_encoder.load_state_dict(pred_enc_dict, strict=False)
            print(f"Loaded temporal encoder weights from {checkpoint_path}")

        # Load decoder weights
        pred_dec_dict = {k.replace('pred_decoder.', ''): v
                         for k, v in state_dict.items() if 'pred_decoder' in k}
        if pred_dec_dict:
            self.pred_decoder.load_state_dict(pred_dec_dict, strict=False)
            print(f"Loaded temporal decoder weights from {checkpoint_path}")

    def freeze_temporal_branch(self):
        """Freeze temporal branch parameters (commonly used during spatial training)."""
        for param in self.pred_encoder.parameters():
            param.requires_grad = False
        for param in self.pred_decoder.parameters():
            param.requires_grad = False
        print("Temporal branch frozen")

    def unfreeze_temporal_branch(self):
        """Unfreeze temporal branch parameters."""
        for param in self.pred_encoder.parameters():
            param.requires_grad = True
        for param in self.pred_decoder.parameters():
            param.requires_grad = True
        print("Temporal branch unfrozen")

    def freeze_bn(self):
        """Freeze all BatchNorm layers"""
        self.seg_model.freeze_bn()


# =============================================================================
# Factory Function
# =============================================================================

def create_stdeeplabv3plus(pred_enc, pred_dec, num_frame=4,
                           num_classes=1,
                           backbone="resnet50",
                           output_stride=16,
                           input_size=256):
    """
    Factory function to create ST-DeepLabV3+ with attention-based decoder.
    Drop-in replacement for create_stcnn_with_attention.

    Args:
        pred_enc: Pretrained temporal prediction encoder
        pred_dec: Pretrained temporal prediction decoder
        num_frame: Number of frames in temporal sequence (default: 4)
        num_classes: Number of segmentation classes (default: 1)
        backbone: ResNet architecture (default: "resnet50")
        output_stride: Output stride (default: 16)
        input_size: Input image size (default: 256)

    Returns:
        ST-DeepLabV3+ model with attention-based decoder
    """
    net = STDeepLabV3Plus(
        pred_enc=pred_enc,
        pred_dec=pred_dec,
        num_classes=num_classes,
        backbone=backbone,
        output_stride=output_stride,
        input_size=input_size
    )

    print(f"Created ST-DeepLabV3+ with attention-based decoder")
    print(f"Backbone: {backbone}, Output stride: {output_stride}, Input size: {input_size}")

    return net


def load_pretrained_stdeeplabv3plus_weights(net, checkpoint_path, strict=False):
    """
    Load weights from checkpoint into ST-DeepLabV3+ model.

    Args:
        net: ST-DeepLabV3+ model
        checkpoint_path: Path to pretrained checkpoint
        strict: If True, requires exact match

    Returns:
        Missing keys and unexpected keys from loading
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    # Get current model state dict
    model_dict = net.state_dict()

    # Load weights that match
    loaded_dict = {}
    missing_in_checkpoint = []
    shape_mismatches = []

    for k, v in model_dict.items():
        if k in pretrained_dict:
            if v.shape == pretrained_dict[k].shape:
                loaded_dict[k] = pretrained_dict[k]
            else:
                shape_mismatches.append(k)
                print(f"  Shape mismatch for {k}: model={v.shape}, checkpoint={pretrained_dict[k].shape}")
        else:
            missing_in_checkpoint.append(k)

    # Update model with loaded weights
    model_dict.update(loaded_dict)
    net.load_state_dict(model_dict, strict=False)

    print(f"\nLoading summary:")
    print(f"  - Successfully loaded: {len(loaded_dict)} parameters")
    print(f"  - Shape mismatches: {len(shape_mismatches)} parameters")
    print(f"  - Missing in checkpoint: {len(missing_in_checkpoint)} parameters")

    if len(missing_in_checkpoint) > 0:
        print(f"\nNew parameters (randomly initialized):")
        for k in missing_in_checkpoint[:10]:
            print(f"    {k}")
        if len(missing_in_checkpoint) > 10:
            print(f"    ... and {len(missing_in_checkpoint) - 10} more")

    # Check which components loaded successfully
    pred_enc_loaded = sum(1 for k in loaded_dict.keys() if 'pred_encoder' in k)
    pred_dec_loaded = sum(1 for k in loaded_dict.keys() if 'pred_decoder' in k)
    seg_model_loaded = sum(1 for k in loaded_dict.keys() if 'seg_model' in k)

    print(f"\nLoaded by component:")
    print(f"  - pred_encoder: {pred_enc_loaded} parameters")
    print(f"  - pred_decoder: {pred_dec_loaded} parameters")
    print(f"  - seg_model: {seg_model_loaded} parameters")

    print(f"\n✓ Weight loading complete!")

    return missing_in_checkpoint, shape_mismatches


# =============================================================================
# Testing and Examples
# =============================================================================