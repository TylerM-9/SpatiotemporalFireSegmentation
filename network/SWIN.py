from .Swin_Unet.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import torch
import torch.nn as nn
import torch.nn.functional as F

class STCNNSWIN(nn.Module):
    def __init__(self, pred_enc, pred_dec, seg):
        super(STCNNSWIN, self).__init__()
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec
        self.seg = seg

    def forward(self, seq, frame):
        pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
        pred, pred_de_feats = self.pred_decoder(pred_en_feats,return_feature_maps=True)
        pred_feats = pred_de_feats

        for i in range(len(pred_de_feats)):
            pred_feats[i] = (pred_feats[i].detach())

        seg_res = self.seg(frame, pred_feats)

        if isinstance(seg_res,list):
            for i in range(len(seg_res)):
                seg_res[i] = F.interpolate(seg_res[i], size=frame.size()[2:], mode='bilinear', align_corners=False)
        else:
            seg_res = F.upsample(seg_res, size=frame.size()[2:], mode='bilinear', align_corners=False)

        return seg_res,pred


class JointDecoderAttention(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()

        self.Conv3x3Middle = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.Conv3x3Out = nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, input: torch.Tensor, high: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Current decoder feature (BCHW format)
            high: Skip connection from encoder (BCHW format)
            temporal: Temporal feature from previous frame (BCHW format)
        Returns:
            Fused features (BCHW format)
        """
        # Concatenate spatial features first
        out = torch.cat([input, high], dim=1)

        # Upsample temporal to match 'out' spatial dimensions
        # Use interpolate instead of Upsample for explicit size matching
        temporal = torch.nn.functional.interpolate(
            temporal,
            size=(out.shape[2], out.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        # Now concatenate temporal with spatial features
        out = torch.cat([temporal, out], dim=1)
        out = self.Conv3x3Middle(out)
        out = self.Conv3x3Out(out)

        return out


class SwinSpatioTemporal(SwinTransformerSys):
    def __init__(self, img_size=256, temporal_channels=[512, 256, 64], *args, **kwargs):
        super().__init__(img_size=img_size, *args, **kwargs)

        # Calculate spatial resolution at each decoder stage
        # Stage 0 (bottleneck): 8x8 -> 16x16, reduces from 768 to 384
        # Stage 1: 16x16 -> 384 (from stage 0) + 384 (skip) + 512 (temporal) = 1280
        # Stage 2: 32x32 -> 192 (from stage 1) + 192 (skip) + 256 (temporal) = 640
        # Stage 3: 64x64 -> 96 (from stage 2) + 96 (skip) + 64 (temporal) = 256
        self.img_size = img_size

        # Temporal attention modules for decoder stages 1, 2, 3
        self.temp_attention = nn.ModuleList([
            JointDecoderAttention(
                n_channels=384 + 384 + temporal_channels[0],  # 1280 channels
                out_channels=384
            ),  # Stage 1: 16x16
            JointDecoderAttention(
                n_channels=192 + 192 + temporal_channels[1],  # 640 channels
                out_channels=192
            ),  # Stage 2: 32x32
            JointDecoderAttention(
                n_channels=96 + 96 + temporal_channels[2],  # 256 channels
                out_channels=96
            ),  # Stage 3: 64x64
        ])

    def blc_to_bchw(self, x, H, W):
        """Convert from BLC to BCHW format"""
        B, L, C = x.shape
        assert L == H * W, f"Mismatch: L={L}, H*W={H * W}"
        x = x.transpose(1, 2).contiguous()  # B, C, L
        x = x.view(B, C, H, W)  # B, C, H, W
        return x

    def bchw_to_blc(self, x):
        """Convert from BCHW to BLC format"""
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # B, C, L
        x = x.transpose(1, 2).contiguous()  # B, L, C
        return x

    def forward_up_features(self, x, x_downsample, x_temp=None):
        """
        Modified decoder with temporal feature fusion

        Args:
            x: Bottleneck features (BLC format)
            x_downsample: List of encoder skip connections (BLC format)
            x_temp: List of temporal features from previous frame (BCHW format)
                    Should have 3 elements for stages 1, 2, 3
        """
        # Calculate spatial resolutions for each stage
        # img_size=256: [8, 16, 32, 64]
        resolutions = [self.img_size // (32 // (2 ** i)) for i in range(4)]

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Stage 0: Just upsample bottleneck (no skip, no temporal)
                x = layer_up(x)  # Still in BLC format
            else:
                # Get current spatial resolution
                H = W = resolutions[inx]

                # Convert current features from BLC to BCHW
                x_bchw = self.blc_to_bchw(x, H, W)

                # Get skip connection and convert to BCHW
                skip_blc = x_downsample[self.num_layers - 1 - inx]
                skip_bchw = self.blc_to_bchw(skip_blc, H, W)

                # Apply temporal attention fusion (all in BCHW format)
                if x_temp is not None and len(x_temp) >= inx:
                    temp_feature = x_temp[inx - 1]  # Already in BCHW format

                    # Debug: print shapes at each stage


                    # JointDecoderAttention handles concat + fusion + dimension reduction
                    x_fused = self.temp_attention[inx - 1](x_bchw, skip_bchw, temp_feature)

                    # Convert back to BLC format - x_fused already has correct output channels
                    x = self.bchw_to_blc(x_fused)

                    # Skip concat_back_dim since JointDecoderAttention already did the reduction
                    # Apply transformer blocks directly
                    x = layer_up(x)
                else:
                    # No temporal features, use original Swin-UNET pathway
                    # Concatenate in BLC format
                    x = torch.cat([x, skip_blc], -1)
                    # Apply dimension reduction
                    x = self.concat_back_dim[inx](x)
                    # Apply transformer blocks
                    x = layer_up(x)

        x = self.norm_up(x)  # Still in BLC format
        return x

    def forward(self, x, x_temp=None):
        """
        Forward pass with optional temporal features

        Args:
            x: Current frame (B, 3, H, W)
            x_temp: List of temporal features from previous decoder stages (BCHW format)
                    Should be [temp_stage1, temp_stage2, temp_stage3]
                    Shapes: [(B, 384, 16, 16), (B, 192, 32, 32), (B, 96, 64, 64)]
        Returns:
            Segmentation output (B, num_classes, H, W)
        """
        # Encoder forward
        x, x_downsample = self.forward_features(x)

        # Decoder forward with temporal features
        x = self.forward_up_features(x, x_downsample, x_temp)

        # Final upsampling to output resolution
        x = self.up_x4(x)

        return x