"""
Clean Spatio-Temporal UNet Architecture
Standard UNet with SimpleContextAdd attention for temporal integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    """Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Features from decoder (upsampled)
            x2: Features from encoder (skip connection)
        """
        x1 = self.up(x1)

        # Handle size mismatch
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================================
# ATTENTION MODULE (from ResUNet)
# ============================================================================

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


# ============================================================================
# UNET ENCODER
# ============================================================================

class UNetEncoder(nn.Module):
    """
    Standard UNet Encoder
    4 downsampling stages: 64 -> 128 -> 256 -> 512
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # Bottleneck

        self.out_channels = [in_channels, 64, 128, 256, 512, 512]

        print(f"UNet Encoder initialized")
        print(f"Encoder output channels: {self.out_channels}")

    def forward(self, x):
        """
        Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: List of [B, C_i, H_i, W_i] tensors at different scales
                     [input, x1, x2, x3, x4, x5]
        """
        x1 = self.inc(x)  # 64 channels, H x W
        x2 = self.down1(x1)  # 128 channels, H/2 x W/2
        x3 = self.down2(x2)  # 256 channels, H/4 x W/4
        x4 = self.down3(x3)  # 512 channels, H/8 x W/8
        x5 = self.down4(x4)  # 512 channels (bottleneck), H/16 x W/16

        return [x, x1, x2, x3, x4, x5]


# ============================================================================
# UNET DECODER WITH ATTENTION
# ============================================================================

class UNetDecoder(nn.Module):
    """
    UNet Decoder with attention modules for temporal integration.
    3 upsampling blocks with optional attention before each.
    """

    def __init__(self, encoder_channels, n_classes=1, use_attention=True):
        """
        Args:
            encoder_channels: List of encoder output channels [3, 64, 128, 256, 512, 512]
            n_classes: Number of output classes
            use_attention: Whether to use attention modules
        """
        super().__init__()

        self.use_attention = use_attention

        # Reverse encoder channels (bottom-up)
        encoder_channels = encoder_channels[::-1]  # [512, 512, 256, 128, 64, 3]

        # Create attention modules for each decoder stage (if enabled)
        if self.use_attention:
            self.attention_blocks = nn.ModuleList()

            # Attention for stage 1: 512 channels
            # Attention for stage 1: x has 512 channels (before up1)
            # Stage 1 attention (before up1): x is 512, temporal/context is 512
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=512,  # x
                    context_channels=512,  # temporal/context @ stage 1
                    temporal_channels=512
                )
            )

            # Stage 2 attention (after up1, before up2): x is 512, temporal/context is 256
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=512,  # x
                    context_channels=256,  # temporal/context @ stage 2
                    temporal_channels=256
                )
            )

            # Stage 3 attention (after up2, before up3): x is 256, temporal/context is 64
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=256,  # x
                    context_channels=64,  # <-- temporal/context @ stage 3 (your crash showed 64)
                    temporal_channels=64
                )
            )
        # Create upsampling blocks
        self.up1 = Up(1024, 512)  # 512 + 512 -> 512
        self.up2 = Up(768, 256)  # 512 + 256 -> 256
        self.up3 = Up(384, 128)  # 256 + 128 -> 128
        self.up4 = Up(192, 64)  # 128 + 64 -> 64

        # Final output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        print(f"UNet Decoder initialized with 4 upsampling blocks")
        print(f"Attention modules: {'Enabled' if use_attention else 'Disabled'}")

    def forward(self, features, temporal_features=None, prev_features=None):
        """
        Forward pass through decoder with optional temporal integration.

        Args:
            features: List of encoder features [x, x1, x2, x3, x4, x5]
            temporal_features: List of temporal decoder features (optional)
                              Should be [temp_512, temp_256, temp_128]
            prev_features: List of previous frame features (optional)
                          Should be [prev_512, prev_256, prev_128]

        Returns:
            x: Segmentation logits [B, n_classes, H, W]
            attention_outputs: List of intermediate attention outputs
        """
        # Reverse features for bottom-up processing
        features = features[::-1]  # [x5, x4, x3, x2, x1, x]

        attention_outputs = []

        # Start from bottleneck
        x = features[0]  # x5 (512 channels)

        # Upsampling stage 1: 512 -> 512
        if self.use_attention and temporal_features is not None and len(temporal_features) > 0:
            temp_feat = temporal_features[0] if 0 < len(temporal_features) else None
            prev_feat = prev_features[0] if prev_features is not None and 0 < len(prev_features) else None

            if temp_feat is not None:
                if prev_feat is None:
                    prev_feat = torch.zeros_like(x)

                _, x = self.attention_blocks[0](
                    x=x,
                    prev=prev_feat,
                    temporal=temp_feat,
                    context_high=temp_feat
                )
                attention_outputs.append(x)

        x = self.up1(x, features[1])  # Up with skip from x4

        # Upsampling stage 2: 512 -> 256
        if self.use_attention and temporal_features is not None and len(temporal_features) > 1:
            temp_feat = temporal_features[1] if 1 < len(temporal_features) else None
            prev_feat = prev_features[1] if prev_features is not None and 1 < len(prev_features) else None

            if temp_feat is not None:
                if prev_feat is None:
                    prev_feat = torch.zeros_like(x)

                _, x = self.attention_blocks[1](
                    x=x,
                    prev=prev_feat,
                    temporal=temp_feat,
                    context_high=temp_feat
                )
                attention_outputs.append(x)

        x = self.up2(x, features[2])  # Up with skip from x3

        # Upsampling stage 3: 256 -> 128
        if self.use_attention and temporal_features is not None and len(temporal_features) > 2:
            temp_feat = temporal_features[2] if 2 < len(temporal_features) else None
            prev_feat = prev_features[2] if prev_features is not None and 2 < len(prev_features) else None

            if temp_feat is not None:
                if prev_feat is None:
                    prev_feat = torch.zeros_like(x)

                _, x = self.attention_blocks[2](
                    x=x,
                    prev=prev_feat,
                    temporal=temp_feat,
                    context_high=temp_feat
                )
                attention_outputs.append(x)

        x = self.up3(x, features[3])  # Up with skip from x2

        # Upsampling stage 4: 128 -> 64 (no attention)
        x = self.up4(x, features[4])  # Up with skip from x1

        # Final output
        logits = self.outc(x)

        return logits, attention_outputs


# ============================================================================
# COMPLETE MODELS
# ============================================================================

class UNet(nn.Module):
    """
    Complete UNet model with encoder and decoder.
    Can be extended with temporal branches.
    """

    def __init__(self, in_channels=3, n_classes=1, use_attention=False):
        super().__init__()

        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_classes=n_classes,
            use_attention=use_attention
        )

    def forward(self, x, temporal_features=None, prev_features=None):
        """
        Forward pass through complete model.

        Args:
            x: Input tensor [B, C, H, W]
            temporal_features: Optional temporal branch features
            prev_features: Optional previous frame features

        Returns:
            segmentation: Logits [B, n_classes, H, W]
            attention_outputs: Intermediate attention outputs
        """
        features = self.encoder(x)
        segmentation, attention_outputs = self.decoder(
            features,
            temporal_features=temporal_features,
            prev_features=prev_features
        )
        return segmentation, attention_outputs


class STUNet(nn.Module):
    """
    Spatio-Temporal UNet
    Integrates temporal prediction branch with spatial segmentation branch.

    Architecture:
    - pred_encoder: Temporal coherence encoder (processes frame sequence)
    - pred_decoder: Temporal coherence decoder (generates predicted frame and features)
    - seg_encoder: Spatial segmentation encoder (UNetEncoder)
    - seg_decoder: Spatial segmentation decoder with attention (UNetDecoder)
    """

    def __init__(self, pred_enc, pred_dec, seg_enc, seg_dec):
        """
        Args:
            pred_enc: Temporal prediction encoder (with pretrained weights)
            pred_dec: Temporal prediction decoder (with pretrained weights)
            seg_enc: Spatial segmentation encoder (UNetEncoder)
            seg_dec: Spatial segmentation decoder (UNetDecoder with attention)
        """
        super().__init__()
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec
        self.seg_encoder = seg_enc
        self.seg_decoder = seg_dec

        print("ST-UNet initialized with temporal and spatial branches")

    def forward(self, seq, frame):
        """
        Forward pass through complete spatio-temporal network.

        Args:
            seq: Sequence of previous frames [B, T*C, H, W]
            frame: Current frame [B, C, H, W]

        Returns:
            seg_res: Segmentation result (logits, attention_outputs)
            pred: Predicted frame from temporal branch [B, C, H, W]
        """
        # === TEMPORAL BRANCH ===
        # Extract temporal features from sequence
        pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)

        # Decode temporal features
        pred, pred_de_feats = self.pred_decoder(pred_en_feats, return_feature_maps=True)

        # Detach temporal features to prevent gradients flowing back
        pred_feats = []
        for feat in pred_de_feats:
            pred_feats.append(feat.detach())

        # === SPATIAL BRANCH ===
        # Extract spatial features from current frame
        seg_en_feats = self.seg_encoder(frame)

        # Decode with attention integration from temporal branch
        seg_logits, attention_outs = self.seg_decoder(seg_en_feats, pred_feats)

        # Upsample to match input size
        seg_logits = F.interpolate(
            seg_logits,
            size=frame.size()[2:],
            mode='bilinear',
            align_corners=False
        )

        # Also upsample attention outputs
        attention_outs_upsampled = []
        for att_out in attention_outs:
            attention_outs_upsampled.append(
                F.interpolate(
                    att_out,
                    size=frame.size()[2:],
                    mode='bilinear',
                    align_corners=False
                )
            )

        return (seg_logits, attention_outs_upsampled), pred

    def freeze_temporal_branch(self):
        """Freeze temporal branch parameters."""
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


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_stunet_with_attention(pred_enc, pred_dec, num_frame=4, n_classes=1):
    """
    Factory function to create ST-UNet with attention-based decoder.

    Args:
        pred_enc: Pretrained temporal prediction encoder
        pred_dec: Pretrained temporal prediction decoder
        num_frame: Number of frames in temporal sequence (default: 4)
        n_classes: Number of segmentation classes (default: 1)

    Returns:
        ST-UNet model with attention-based decoder
    """
    # Create spatial segmentation encoder
    seg_enc = UNetEncoder(in_channels=3)

    # Create spatial segmentation decoder with attention
    seg_dec = UNetDecoder(
        encoder_channels=seg_enc.out_channels,
        n_classes=n_classes,
        use_attention=True
    )

    # Create full ST-UNet model
    net = STUNet(
        pred_enc=pred_enc,
        pred_dec=pred_dec,
        seg_enc=seg_enc,
        seg_dec=seg_dec
    )

    print(f"Created ST-UNet with attention-based decoder")
    print(f"Input channels: 3, Output classes: {n_classes}")

    return net


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing ST-UNet architecture...\n")

    # Test standard UNet without attention
    print("=" * 60)
    print("Test 1: Standard UNet (no attention)")
    print("=" * 60)
    model = UNet(in_channels=3, n_classes=1, use_attention=False)
    x = torch.randn(2, 3, 256, 256)
    output, attn = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}\n")

    # Test UNet with attention (no temporal features)
    print("=" * 60)
    print("Test 2: UNet with attention (no temporal)")
    print("=" * 60)
    model = UNet(in_channels=3, n_classes=1, use_attention=True)
    output, attn = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}\n")

    # Test UNet with attention and temporal features
    print("=" * 60)
    print("Test 3: UNet with attention and temporal features")
    print("=" * 60)
    temporal_feats = [
        torch.randn(2, 512, 16, 16),  # Bottleneck
        torch.randn(2, 512, 32, 32),  # Stage 1
        torch.randn(2, 256, 64, 64),  # Stage 2
    ]
    output, attn = model(x, temporal_features=temporal_feats)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}")
    for i, a in enumerate(attn):
        print(f"  Attention {i + 1}: {a.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n✓ All tests passed!")