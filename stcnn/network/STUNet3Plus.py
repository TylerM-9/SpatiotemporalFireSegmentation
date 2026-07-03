"""
Spatio-Temporal UNet3Plus Architecture
Advanced UNet3Plus with SimpleContextAdd attention for temporal integration
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

class UNet3PlusEncoder(nn.Module):
    """
    Standard UNet3Plus Encoder
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

        print(f"UNet3Plus Encoder initialized")
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
# UNet3Plus DECODER WITH ATTENTION
# ============================================================================

class UNet3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, n_classes: int = 1, use_attention: bool = True):
        """
        Args:
            encoder_channels: List of encoder output channels [3, 64, 128, 256, 512, 512]
            n_classes: Number of output classes
            use_attention: Whether to use attention modules
        """
        super().__init__()
        
        self.use_attention = use_attention
        target_ch = 64  # Every branch scales its output down to 64 channels
        unet3plus_concat_channels = target_ch * 5  # 5 branches * 64 = 320 channels total

        # =================================================================
        # 1. DECODER LEVEL 4 (d4) ROUTING (Target: H/8, W/8)
        # =================================================================
        self.e1_to_d4 = nn.Sequential(nn.MaxPool2d(8, stride=8), nn.Conv2d(64, target_ch, 3, padding=1, bias=False))
        self.e2_to_d4 = nn.Sequential(nn.MaxPool2d(4, stride=4), nn.Conv2d(128, target_ch, 3, padding=1, bias=False))
        self.e3_to_d4 = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(256, target_ch, 3, padding=1, bias=False))
        self.e4_to_d4 = nn.Sequential(nn.Conv2d(512, target_ch, 3, padding=1, bias=False))
        self.e5_to_d4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(512, target_ch, 3, padding=1, bias=False))

        # Output convolution to merge concatenated states
        self.conv_d4 = DoubleConv(unet3plus_concat_channels, unet3plus_concat_channels)

        if self.use_attention:
            # d4 is H/8. It gets pred_feats[1], which has 256 channels.
            self.sta4 = SimpleContextAdd(in_channels=unet3plus_concat_channels, context_channels=256, temporal_channels=256)
        self.compress_d4 = nn.Conv2d(unet3plus_concat_channels, target_ch, kernel_size=1, bias=False)

        # =================================================================
        # 2. DECODER LEVEL 3 (d3) ROUTING (Target: H/4, W/4)
        # =================================================================
        self.e1_to_d3 = nn.Sequential(nn.MaxPool2d(4, stride=4), nn.Conv2d(64, target_ch, 3, padding=1, bias=False))
        self.e2_to_d3 = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(128, target_ch, 3, padding=1, bias=False))
        self.e3_to_d3 = nn.Sequential(nn.Conv2d(256, target_ch, 3, padding=1, bias=False))
        self.d4_to_d3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.e5_to_d3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(512, target_ch, 3, padding=1, bias=False))

        self.conv_d3 = DoubleConv(unet3plus_concat_channels, unet3plus_concat_channels)

        if self.use_attention:
            # d3 is H/4. It gets pred_feats[2], which has 64 channels.
            self.sta3 = SimpleContextAdd(in_channels=unet3plus_concat_channels, context_channels=64, temporal_channels=64)
        self.compress_d3 = nn.Conv2d(unet3plus_concat_channels, target_ch, kernel_size=1, bias=False)

        # =================================================================
        # 3. DECODER LEVEL 2 (d2) ROUTING (Target: H/2, W/2)
        # =================================================================
        self.e1_to_d2 = nn.Sequential(nn.MaxPool2d(2, stride=2), nn.Conv2d(64, target_ch, 3, padding=1, bias=False))
        self.e2_to_d2 = nn.Sequential(nn.Conv2d(128, target_ch, 3, padding=1, bias=False))
        self.d3_to_d2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.d4_to_d2 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.e5_to_d2 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False), nn.Conv2d(512, target_ch, 3, padding=1, bias=False))

        self.conv_d2 = DoubleConv(unet3plus_concat_channels, unet3plus_concat_channels)

        if self.use_attention:
            # d2 is H/2. We will upsample pred_feats[2] for it, so it is still 64 channels.
            self.sta2 = SimpleContextAdd(in_channels=unet3plus_concat_channels, context_channels=64, temporal_channels=64)
        self.compress_d2 = nn.Conv2d(unet3plus_concat_channels, target_ch, kernel_size=1, bias=False)

        # =================================================================
        # 4. DECODER LEVEL 1 (d1) ROUTING (Target: H, W)
        # =================================================================
        self.e1_to_d1 = nn.Sequential(nn.Conv2d(64, target_ch, 3, padding=1, bias=False))
        self.d2_to_d1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.d3_to_d1 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.d4_to_d1 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False), nn.Conv2d(target_ch, target_ch, 3, padding=1, bias=False))
        self.e5_to_d1 = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False), nn.Conv2d(512, target_ch, 3, padding=1, bias=False))

        self.conv_d1 = DoubleConv(unet3plus_concat_channels, unet3plus_concat_channels)

        # Final output layer
        self.outc = nn.Conv2d(unet3plus_concat_channels, n_classes, kernel_size=1)

        print(f"UNet3Plus Decoder initialized with 4 unified routing blocks")


    def forward(self, features, temporal_features=None, prev_features=None):
        # Keep native indexing: features = [input, x1, x2, x3, x4, x5]
        _, x1, x2, x3, x4, x5 = features

        attention_outputs = []

        # Level 5 is the bottleneck itself
        d5 = x5 

        # =================================================================
        # DECODER LEVEL 4
        # =================================================================
        d4 = torch.cat([
            self.e1_to_d4(x1), self.e2_to_d4(x2),
            self.e3_to_d4(x3), self.e4_to_d4(x4), self.e5_to_d4(d5)
        ], dim=1)
        d4 = self.conv_d4(d4)

        if self.use_attention and temporal_features is not None and len(temporal_features) > 0:
            temp_feat = temporal_features[0]
            prev_feat = prev_features[0] if (prev_features is not None and len(prev_features) > 0) else torch.zeros_like(d4)
            _, d4 = self.sta4(x=d4, prev=prev_feat, temporal=temp_feat, context_high=temp_feat)
            attention_outputs.append(d4)
        
        # Compress from 320 -> 64 channels so d3, d2, d1 blocks can ingest it
        d4_compressed = self.compress_d4(d4)

        # =================================================================
        # DECODER LEVEL 3
        # =================================================================
        d3 = torch.cat([
            self.e1_to_d3(x1), self.e2_to_d3(x2),
            self.e3_to_d3(x3), self.d4_to_d3(d4_compressed), self.e5_to_d3(d5)
        ], dim=1)
        d3 = self.conv_d3(d3)

        if self.use_attention and temporal_features is not None and len(temporal_features) > 1:
            temp_feat = temporal_features[1]
            prev_feat = prev_features[1] if (prev_features is not None and len(prev_features) > 1) else torch.zeros_like(d3)
            _, d3 = self.sta3(x=d3, prev=prev_feat, temporal=temp_feat, context_high=temp_feat)
            attention_outputs.append(d3)
            
        d3_compressed = self.compress_d3(d3)

        # =================================================================
        # DECODER LEVEL 2
        # =================================================================
        d2 = torch.cat([
            self.e1_to_d2(x1), self.e2_to_d2(x2),
            self.d3_to_d2(d3_compressed), self.d4_to_d2(d4_compressed), self.e5_to_d2(d5)
        ], dim=1)
        d2 = self.conv_d2(d2)

        if self.use_attention and temporal_features is not None and len(temporal_features) > 2:
            temp_feat = temporal_features[2]
            prev_feat = prev_features[2] if (prev_features is not None and len(prev_features) > 2) else torch.zeros_like(d2)
            _, d2 = self.sta2(x=d2, prev=prev_feat, temporal=temp_feat, context_high=temp_feat)
            attention_outputs.append(d2)
            
        d2_compressed = self.compress_d2(d2)

        # =================================================================
        # DECODER LEVEL 1
        # =================================================================
        d1 = torch.cat([
            self.e1_to_d1(x1), self.d2_to_d1(d2_compressed),
            self.d3_to_d1(d3_compressed), self.d4_to_d1(d4_compressed), self.e5_to_d1(d5)
        ], dim=1)
        d1 = self.conv_d1(d1)

        # Final prediction layer out of d1
        logits = self.outc(d1)

        return logits, attention_outputs


# ============================================================================
# COMPLETE MODELS
# ============================================================================

class UNet3Plus(nn.Module):
    """
    Complete UNet3Plus model with encoder and full-scale skip routing decoder.
    """
    def __init__(self, in_channels=3, n_classes=1, use_attention=False):
        super().__init__()

        self.encoder = UNet3PlusEncoder(in_channels=in_channels)
        self.decoder = UNet3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            n_classes=n_classes,
            use_attention=use_attention
        )

    def forward(self, x, temporal_features=None, prev_features=None):
        """
        Forward pass through complete model.
        """
        features = self.encoder(x)
        segmentation, attention_outputs = self.decoder(
            features,
            temporal_features=temporal_features,
            prev_features=prev_features
        )
        return segmentation, attention_outputs


class STUNet3Plus(nn.Module):
    """
    Spatio-Temporal UNet3Plus with full routing attention-based decoder.
    """
    def __init__(self, pred_enc, pred_dec, seg_enc, seg_dec):
        super().__init__()
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec
        self.seg_encoder = seg_enc
        self.seg_decoder = seg_dec

        print("STUNet3Plus initialized with temporal and spatial branches")

    def forward(self, seq, frame):
        # === TEMPORAL BRANCH ===
        pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
        pred, pred_de_feats = self.pred_decoder(pred_en_feats, return_feature_maps=True)

        # Detach temporal features to isolate gradients
        pred_feats = [feat.detach() for feat in pred_de_feats]

        # pred_feats[1] -> H/8 (256ch) matches d4
        # pred_feats[2] -> H/4 (64ch) matches d3
        # d2 is H/2, but pred_feats ends at H/4. We must interpolate pred_feats[2] to match d2.
        
        feat_for_d2 = F.interpolate(
            pred_feats[2], 
            scale_factor=2, 
            mode='bilinear', 
            align_corners=False
        )

        aligned_temporal_feats = [
            pred_feats[1],  # For d4 (H/8)
            pred_feats[2],  # For d3 (H/4)
            feat_for_d2     # For d2 (H/2)
        ]

        # === SPATIAL BRANCH ===
        seg_en_feats = self.seg_encoder(frame)

        # Decode with properly aligned attention routing from the temporal branch
        seg_logits, attention_outs = self.seg_decoder(
            features=seg_en_feats, 
            temporal_features=aligned_temporal_feats,
            prev_features=[]  # No previous features provided for attention
        )

        # Upsample logits to match target input resolution if needed
        if seg_logits.shape[2:] != frame.shape[2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=frame.size()[2:],
                mode='bilinear',
                align_corners=False
            )

        # Upsample attention outputs safely
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
        for param in self.pred_encoder.parameters():
            param.requires_grad = False
        for param in self.pred_decoder.parameters():
            param.requires_grad = False
        print("Temporal branch frozen")

    def unfreeze_temporal_branch(self):
        for param in self.pred_encoder.parameters():
            param.requires_grad = True
        for param in self.pred_decoder.parameters():
            param.requires_grad = True
        print("Temporal branch unfrozen")


# ============================================================================
# FACTORY FUNCTION FOR UNET3PLUS
# ============================================================================

def create_unet3plus(pred_enc, pred_dec, num_frame=4, n_classes=1):
    """
    Factory function to create STUNet3Plus with attention-based decoder.
    """
    # Create spatial segmentation encoder
    seg_enc = UNet3PlusEncoder(in_channels=3)

    # Create spatial segmentation decoder with attention
    seg_dec = UNet3PlusDecoder(
        encoder_channels=seg_enc.out_channels,
        n_classes=n_classes,
        use_attention=True
    )

    # Create full STUNet3Plus model
    net = STUNet3Plus(
        pred_enc=pred_enc,
        pred_dec=pred_dec,
        seg_enc=seg_enc,
        seg_dec=seg_dec
    )

    print(f"Created STUNet3Plus with full-scale skip attention decoder")
    print(f"Input channels: 3, Output classes: {n_classes}")

    return net


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing ST-UNet3Plus architecture...\n")

    # Mock Input Setup
    x = torch.randn(2, 3, 256, 256)

    # Test 1: UNet3Plus without attention
    print("=" * 60)
    print("Test 1: Advanced UNet3Plus (no attention)")
    print("=" * 60)
    model_no_attn = UNet3Plus(in_channels=3, n_classes=1, use_attention=False)
    output, attn = model_no_attn(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}\n")

    # Test 2: UNet3Plus with attention (no temporal features provided)
    print("=" * 60)
    print("Test 2: Advanced UNet3Plus with attention (no temporal)")
    print("=" * 60)
    model_attn = UNet3Plus(in_channels=3, n_classes=1, use_attention=True)
    output, attn = model_attn(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}\n")

    # Test 3: UNet3Plus with attention AND active temporal features
    print("=" * 60)
    print("Test 3: Advanced UNet3Plus with attention and temporal features")
    print("=" * 60)
    
    # Matching your fixed decoder definitions:
    # sta4 expects context_channels=512, temporal_channels=512
    # sta3 expects context_channels=512, temporal_channels=512
    # sta2 expects context_channels=256, temporal_channels=256
    temporal_feats = [
        torch.randn(2, 512, 32, 32),  # Target scale for Level 4 (d4) -> H/8, W/8
        torch.randn(2, 512, 64, 64),  # Target scale for Level 3 (d3) -> H/4, W/4
        torch.randn(2, 256, 128, 128) # Target scale for Level 2 (d2) -> H/2, W/2
    ]
    
    # We create matching mock prev_features inside the same channel dimension bounds 
    # to avoid multiplication shape crashes inside SimpleContextAdd step 4
    prev_feats = [
        torch.randn(2, 320, 32, 32),   # Matches d4 output channel space (320)
        torch.randn(2, 320, 64, 64),   # Matches d3 output channel space (320)
        torch.randn(2, 320, 128, 128)  # Matches d2 output channel space (320)
    ]

    output, attn = model_attn(x, temporal_features=temporal_feats, prev_features=prev_feats)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Attention outputs: {len(attn)}")
    for i, a in enumerate(attn):
        print(f"  Attention Stage {i + 4}: {a.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model_attn.parameters())
    trainable_params = sum(p.numel() for p in model_attn.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n✓ All UNet3+ pipeline tests passed successfully!")

    