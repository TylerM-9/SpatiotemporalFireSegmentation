"""
Complete Spatio-Temporal CNN Architecture
Includes all necessary components for video segmentation with temporal coherence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ============================================================================
# ENCODER
# ============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet encoder extracted from segmentation_models_pytorch.
    Returns multi-scale features for skip connections.
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3):
        super(ResNetEncoder, self).__init__()

        # Create a temporary Unet model to extract the encoder
        temp_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1
        )

        # Extract the encoder
        self.encoder = temp_model.encoder
        self.encoder_name = encoder_name
        self.out_channels = self.encoder.out_channels  # e.g., [3, 64, 64, 128, 256, 512] for resnet34

        print(f"Encoder initialized: {encoder_name}")
        print(f"Encoder output channels: {self.out_channels}")

    def forward(self, x, return_feature_maps=False):
        """
        Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W]
            return_feature_maps: If True, return list of features (for compatibility)

        Returns:
            features: List of [B, C_i, H_i, W_i] tensors at different scales
                     e.g., for resnet34: [x, conv1, layer1, layer2, layer3, layer4]
        """
        features = self.encoder(x)
        return features

    def load_pretrained_weights(self, checkpoint_path):
        """Load encoder weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter encoder weights
        encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k}
        self.encoder.load_state_dict(encoder_dict, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")


# ============================================================================
# ATTENTION MODULE
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

        # If temporal_channels not specified, assume same as context_channels
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
            in_channels * 2,  # concat(out_down, temporal) - both have in_channels after projection
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        # Second 3x3 conv: final processing
        # Input is concat(out_forward1, out_forward2) where:
        # - out_forward1 has in_channels * 2
        # - out_forward2 has in_channels
        # Total: in_channels * 3
        self.conv3x3_2 = nn.Conv2d(
            in_channels * 3,  # concat(out_forward1[2*C], out_forward2[C])
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
        out_forward2 = middle_conv * prev

        # Step 5: Concatenate both branches
        out = torch.cat((out_forward1, out_forward2), dim=1)

        # Step 6: Final convolution
        out = self.conv3x3_2(out)

        return out_down, out


# ============================================================================
# DECODER COMPONENTS
# ============================================================================

class DecoderBlock(nn.Module):
    """
    Single decoder block: Upsample -> Concat Skip -> Conv -> Conv
    Standard UNet decoder block with skip connections.
    """

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Adjust input channels based on whether we have skip connection
        conv_in_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetDecoder(nn.Module):
    """
    UNet-style decoder with attention modules and 3 upsampling blocks.
    Integrates temporal and context information via SimpleContextAdd before each decoder block.
    """

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64), n_classes=1,
                 use_attention=True):
        """
        Args:
            encoder_channels: List of encoder output channels, e.g., [3, 64, 64, 128, 256, 512]
            decoder_channels: Tuple of 3 decoder channels (256, 128, 64) to match diagram
            n_classes: Number of output classes
            use_attention: Whether to use attention modules (for temporal integration)
        """
        super(ResNetDecoder, self).__init__()

        assert len(decoder_channels) == 3, "Decoder must have exactly 3 blocks to match architecture"

        self.use_attention = use_attention

        # Reverse encoder channels (bottom-up)
        # For ResNet34: [3, 64, 64, 128, 256, 512] -> [512, 256, 128, 64, 64, 3]
        encoder_channels = encoder_channels[::-1]

        # Extract the channels we'll use for skip connections
        # encoder_channels[0] = 512 (bottleneck, res5)
        # encoder_channels[1] = 256 (res4)
        # encoder_channels[2] = 128 (res3)
        # encoder_channels[3] = 64  (res2)

        # Create attention modules for each decoder stage (if enabled)
        if self.use_attention:
            self.attention_blocks = nn.ModuleList()

            # Attention for Block 1: integrates features at 256-channel level
            # Context and temporal both come from temporal decoder
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=decoder_channels[0],      # 256 (current decoder stage)
                    context_channels=encoder_channels[0],  # 512 (from temporal bottleneck)
                    temporal_channels=encoder_channels[0]  # 512 (from temporal bottleneck)
                )
            )

            # Attention for Block 2: integrates features at 128-channel level
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=decoder_channels[1],      # 128
                    context_channels=decoder_channels[0],  # 256 from temporal decoder stage 1
                    temporal_channels=decoder_channels[0]  # 256 from temporal decoder stage 1
                )
            )

            # Attention for Block 3: integrates features at 64-channel level
            self.attention_blocks.append(
                SimpleContextAdd(
                    in_channels=decoder_channels[2],      # 64
                    context_channels=decoder_channels[1],  # 128 from temporal decoder stage 2
                    temporal_channels=decoder_channels[1]  # 128 from temporal decoder stage 2
                )
            )

        # Create 3 decoder blocks matching the diagram
        self.blocks = nn.ModuleList()

        # Block 1: 512 -> 256 (with skip from res4: 256 channels)
        self.blocks.append(
            DecoderBlock(
                in_channels=encoder_channels[0],  # 512 from res5
                out_channels=decoder_channels[0],  # 256
                skip_channels=encoder_channels[1]  # 256 from res4
            )
        )

        # Block 2: 256 -> 128 (with skip from res3: 128 channels)
        self.blocks.append(
            DecoderBlock(
                in_channels=decoder_channels[0],  # 256 from previous block
                out_channels=decoder_channels[1],  # 128
                skip_channels=encoder_channels[2]  # 128 from res3
            )
        )

        # Block 3: 128 -> 64 (with skip from res2: 64 channels)
        self.blocks.append(
            DecoderBlock(
                in_channels=decoder_channels[1],  # 128 from previous block
                out_channels=decoder_channels[2],  # 64
                skip_channels=encoder_channels[3]  # 64 from res2
            )
        )

        # Final segmentation head: 64 -> n_classes
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            n_classes,
            kernel_size=3,
            padding=1
        )

        print(f"Decoder initialized with 3 blocks: {decoder_channels}")
        print(f"Attention modules: {'Enabled' if use_attention else 'Disabled'}")
        print(f"Skip connections from encoder channels: [{encoder_channels[1]}, {encoder_channels[2]}, {encoder_channels[3]}]")

    def forward(self, features, temporal_features=None, prev_features=None):
        """
        Forward pass through decoder with optional temporal integration.

        Args:
            features: List of encoder features [f0, f1, f2, f3, f4, f5]
                     where f0 is input, f5 is bottleneck
            temporal_features: List of temporal decoder features at each scale (optional)
                              Should match decoder stages: [temp_256, temp_128, temp_64]
            prev_features: List of previous frame features at each scale (optional)
                          Should match decoder stages: [prev_256, prev_128, prev_64]

        Returns:
            x: Segmentation logits [B, n_classes, H, W]
            attention_outputs: List of intermediate attention outputs (for loss computation)
        """
        # Reverse features for bottom-up processing
        features = features[::-1]  # [bottleneck, ..., input]

        # Initialize attention outputs storage
        attention_outputs = []

        # First decoder block (process bottleneck)
        x = self.blocks[0](features[0], features[1])  # Pass bottleneck and first skip

        # Process subsequent blocks with optional attention
        for i in range(1, len(self.blocks)):
            # Get skip connection
            skip = features[i + 1] if (i + 1) < len(features) else None

            # Apply attention if enabled and temporal features provided
            if self.use_attention and temporal_features is not None:
                # Get temporal and previous features for this stage
                temp_feat = temporal_features[i - 1] if i - 1 < len(temporal_features) else None
                prev_feat = prev_features[i - 1] if prev_features is not None and i - 1 < len(prev_features) else None

                if temp_feat is not None:
                    # Use zero features if prev_feat not provided
                    if prev_feat is None:
                        prev_feat = torch.zeros_like(x)

                    # Apply attention module (handles all resizing and projection internally)
                    out_down, x_attended = self.attention_blocks[i - 1](
                        x=x,
                        prev=prev_feat,
                        temporal=temp_feat,
                        context_high=temp_feat  # Using temporal as context
                    )

                    # Store attention output for auxiliary loss
                    attention_outputs.append(x_attended)

                    # Use attended features
                    x = x_attended

            # Apply decoder block
            x = self.blocks[i](x, skip)

        # Final segmentation head
        x = self.segmentation_head(x)

        return x, attention_outputs


# ============================================================================
# COMPLETE MODELS
# ============================================================================

class ResUNet(nn.Module):
    """
    Complete ResUNet model with decomposed encoder and decoder.
    Can be extended with temporal branches.
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, n_classes=1, decoder_channels=(256, 128, 64),
                 use_attention=True):
        super(ResUNet, self).__init__()

        self.encoder = ResNetEncoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )

        self.decoder = ResNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
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

    def get_encoder_features(self, x):
        """
        Get intermediate encoder features (useful for temporal branch).

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: List of multi-scale features
        """
        return self.encoder(x)


class STCNN(nn.Module):
    """
    Spatio-Temporal Convolutional Neural Network
    Integrates temporal prediction branch with spatial segmentation branch.

    Architecture:
    - pred_encoder: Temporal coherence encoder (processes frame sequence t-4 to t-1)
    - pred_decoder: Temporal coherence decoder (generates predicted frame t and features)
    - seg_encoder: Spatial segmentation encoder (processes current frame t)
    - seg_decoder: Spatial segmentation decoder with attention (uses temporal features)
    """

    def __init__(self, pred_enc, seg_enc, pred_dec, seg_dec):
        """
        Args:
            pred_enc: Temporal prediction encoder (with pretrained weights)
            seg_enc: Spatial segmentation encoder (ResNetEncoder)
            pred_dec: Temporal prediction decoder (with pretrained weights)
            seg_dec: Spatial segmentation decoder (ResNetDecoder with attention)
        """
        super(STCNN, self).__init__()
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec
        self.seg_encoder = seg_enc
        self.seg_decoder = seg_dec

        print("STCNN initialized with temporal and spatial branches")

    def forward(self, seq, frame):
        """
        Forward pass through complete spatio-temporal network.

        Args:
            seq: Sequence of previous frames [B, T*C, H, W] where T is temporal window
                 e.g., for 4 frames: [B, 12, H, W] if RGB
            frame: Current frame [B, C, H, W]

        Returns:
            seg_res: Segmentation result (list of outputs at different scales or single output)
            pred: Predicted frame from temporal branch [B, C, H, W]
        """
        # === TEMPORAL BRANCH (TOP) ===
        # Extract temporal features from sequence (t-4, t-3, t-2, t-1)
        pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)

        # Decode temporal features to get prediction and decoder features
        pred, pred_de_feats = self.pred_decoder(pred_en_feats, return_feature_maps=True)

        # Detach temporal features to prevent gradients flowing back to temporal branch
        # This keeps temporal branch frozen during spatial branch training
        pred_feats = []
        for feat in pred_de_feats:
            pred_feats.append(feat.detach())

        # === SPATIAL BRANCH (BOTTOM) ===
        # Extract spatial features from current frame
        seg_en_feats = self.seg_encoder(frame, return_feature_maps=True)

        # Decode with attention integration from temporal branch
        seg_res = self.seg_decoder(seg_en_feats, pred_feats)

        # Upsample segmentation results to match input size
        if isinstance(seg_res, tuple):
            # seg_res is (segmentation, attention_outputs)
            seg_out, attention_outs = seg_res
            seg_out = F.interpolate(
                seg_out,
                size=frame.size()[2:],
                mode='bilinear',
                align_corners=False
            )
            # Also upsample attention outputs for auxiliary losses
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
            seg_res = (seg_out, attention_outs_upsampled)
        elif isinstance(seg_res, list):
            # Multiple outputs at different scales
            for i in range(len(seg_res)):
                seg_res[i] = F.interpolate(
                    seg_res[i],
                    size=frame.size()[2:],
                    mode='bilinear',
                    align_corners=False
                )
        else:
            # Single output
            seg_res = F.interpolate(
                seg_res,
                size=frame.size()[2:],
                mode='bilinear',
                align_corners=False
            )

        return seg_res, pred

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


# ============================================================================
# FACTORY FUNCTION FOR TRAINING SCRIPT
# ============================================================================

def create_stcnn_with_attention(pred_enc, pred_dec, num_frame=4,
                                encoder_name="resnet34",
                                encoder_weights="imagenet",
                                decoder_channels=(256, 128, 64),
                                n_classes=1):
    """
    Factory function to create STCNN with attention-based segmentation decoder.
    This is a drop-in replacement for the original STCNN initialization.

    Args:
        pred_enc: Pretrained temporal prediction encoder
        pred_dec: Pretrained temporal prediction decoder
        num_frame: Number of frames in temporal sequence (default: 4)
        encoder_name: ResNet encoder architecture (default: "resnet34")
        encoder_weights: Pretrained weights for encoder (default: "imagenet")
        decoder_channels: Decoder channel configuration (default: (256, 128, 64))
        n_classes: Number of segmentation classes (default: 1)

    Returns:
        STCNN model with attention-based decoder
    """
    # Create spatial segmentation encoder
    seg_enc = ResNetEncoder(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3
    )

    # Create spatial segmentation decoder with attention
    seg_dec = ResNetDecoder(
        encoder_channels=seg_enc.out_channels,
        decoder_channels=decoder_channels,
        n_classes=n_classes,
        use_attention=True  # Enable attention for temporal integration
    )

    # Create full STCNN model
    net = STCNN(
        pred_enc=pred_enc,
        seg_enc=seg_enc,
        pred_dec=pred_dec,
        seg_dec=seg_dec
    )

    print(f"Created STCNN with attention-based decoder")
    print(f"Encoder: {encoder_name}, Decoder channels: {decoder_channels}")

    return net


def load_pretrained_stcnn_weights(net, checkpoint_path, strict=False):
    """
    Load weights from old STCNN model into new attention-based STCNN.

    This function handles weight mapping between the old architecture and new architecture.
    The temporal branch (pred_encoder, pred_decoder) should load directly.
    The spatial encoder (seg_encoder) should load directly if using same ResNet backbone.
    The spatial decoder (seg_decoder) may have mismatches due to attention modules.

    Args:
        net: New STCNN model with attention
        checkpoint_path: Path to pretrained checkpoint
        strict: If True, requires exact match (default: False for flexibility)

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

    # Separate keys by component
    pred_encoder_keys = [k for k in pretrained_dict.keys() if 'pred_encoder' in k]
    pred_decoder_keys = [k for k in pretrained_dict.keys() if 'pred_decoder' in k]
    seg_encoder_keys = [k for k in pretrained_dict.keys() if 'seg_encoder' in k]
    seg_decoder_keys = [k for k in pretrained_dict.keys() if 'seg_decoder' in k]

    print(f"\nFound in checkpoint:")
    print(f"  - pred_encoder keys: {len(pred_encoder_keys)}")
    print(f"  - pred_decoder keys: {len(pred_decoder_keys)}")
    print(f"  - seg_encoder keys: {len(seg_encoder_keys)}")
    print(f"  - seg_decoder keys: {len(seg_decoder_keys)}")

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
    print(f"  - Missing in checkpoint (new params): {len(missing_in_checkpoint)} parameters")

    if len(missing_in_checkpoint) > 0:
        print(f"\nNew parameters (randomly initialized):")
        for k in missing_in_checkpoint[:10]:  # Show first 10
            print(f"    {k}")
        if len(missing_in_checkpoint) > 10:
            print(f"    ... and {len(missing_in_checkpoint) - 10} more")

    # Check which components loaded successfully
    pred_enc_loaded = sum(1 for k in loaded_dict.keys() if 'pred_encoder' in k)
    pred_dec_loaded = sum(1 for k in loaded_dict.keys() if 'pred_decoder' in k)
    seg_enc_loaded = sum(1 for k in loaded_dict.keys() if 'seg_encoder' in k)
    seg_dec_loaded = sum(1 for k in loaded_dict.keys() if 'seg_decoder' in k)

    print(f"\nLoaded by component:")
    print(f"  - pred_encoder: {pred_enc_loaded} parameters")
    print(f"  - pred_decoder: {pred_dec_loaded} parameters")
    print(f"  - seg_encoder: {seg_enc_loaded} parameters")
    print(f"  - seg_decoder: {seg_dec_loaded} parameters")

    print(f"\n✓ Weight loading complete!")

    return missing_in_checkpoint, shape_mismatches