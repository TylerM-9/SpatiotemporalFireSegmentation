import torch.nn as nn

import torch
import torch.nn.functional as F

class STCNN2(nn.Module):
	def __init__(self, pred_enc, seg_enc, pred_dec, j_seg_dec):
		super(STCNN2, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		self.seg_encoder = seg_enc
		self.seg_decoder = j_seg_dec

	def forward(self, seq, frame):
		pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
		pred, pred_de_feats = self.pred_decoder(pred_en_feats,return_feature_maps=True)
		pred_feats = pred_de_feats
		for i in range(len(pred_de_feats)):
			pred_feats[i] = (pred_feats[i].detach())
		seg_en_feats = self.seg_encoder(frame)

		seg_res = self.seg_decoder(seg_en_feats, pred_feats)

		if isinstance(seg_res,list):
			for i in range(len(seg_res)):
				seg_res[i] = F.interpolate(seg_res[i], size=frame.size()[2:], mode='bilinear', align_corners=False)
		else:
			seg_res = F.upsample(seg_res, size=frame.size()[2:], mode='bilinear', align_corners=False)

		return seg_res,pred

class SegBranch(nn.Module):
    def __init__(self, net_enc: nn.Module, net_dec: nn.Module):
        super().__init__()
        self.net_enc = net_enc
        self.net_dec = net_dec

    def forward(self, x):
        features = self.net_enc(x)
        out = self.net_dec(features)
        return out



class SegEncoder(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        self.haarDownsample2x1 = HaarDownsample2x(16)
        self.haarDownsample2x2 = HaarDownsample2x(32)
        self.haarDownsample2x3 = HaarDownsample2x(64)
        self.haarDownsample2x4 = HaarDownsample2x(128)


        self.pre_layer = nn.Sequential(
            ConvBNReLU(num_channels,16)
        )

        self.layer1 = nn.Sequential(
            ConvBNReLU(16,16),
            ConvBNReLU(16,16),
            ResidualBlockWith1x1(16,16),
            CBAM(16)
            
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(32,32),
            ConvBNReLU(32,32),
            ResidualBlockWith1x1(32,32),
            CBAM(32),
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(64,64),
            ConvBNReLU(64,64),
            ResidualBlockWith1x1(64,64),
            CBAM(64),

        )
        self.layer4 = nn.Sequential(
            ConvBNReLU(128,128),
            ConvBNReLU(128,128),
            ResidualBlockWith1x1(128,128),
            CBAM(128),
        )

    def forward(self, x):
        x = self.pre_layer(x)  # (B,16,H,W)
        x1 = self.layer1(x)    # (B,16,H,W)
        x = self.haarDownsample2x1(x1)  # (B,32,H/2,W/2)
        x2 = self.layer2(x)    # (B,32,H/2,W/2)
        x = self.haarDownsample2x2(x2)  # (B,64,H/4,W/4)
        x3 = self.layer3(x)    # (B,64,H/4,W/4)
        x = self.haarDownsample2x3(x3)  # (B,128,H/8,W/8)   
        x4 = self.layer4(x)    # (B,128,H/8,W/8)
        x5 = self.haarDownsample2x4(x4)  # (B,256,H/16,W/16)

        return x1, x2, x3, x4, x5  # return features at multiple scales

class SegDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.ResidualBlockWith1x1_1 = ResidualBlockWith1x1(16,16)
        self.ResidualBlockWith1x1_2 = ResidualBlockWith1x1(32,32)
        self.ResidualBlockWith1x1_3 = ResidualBlockWith1x1(64,64)
        self.ResidualBlockWith1x1_4 = ResidualBlockWith1x1(128,128)
        self.ResidualBlockWith1x1_5 = ResidualBlockWith1x1(256,256)

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

        # Final 1x1 conv to get desired number of classes
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features  # x1: (B,32,H/2,W/2), x2: (B,64,H/4,W/4), x3: (B,128,H/8,W/8), x4: (B,256,H/16,W/16)

        x5 = self.ResidualBlockWith1x1_5(x5)
        x = self.up4(x5)  # (B,128,H/8,W/8)
        x += x4      # Skip connection

        x = self.conv4(x)
        x = self.ResidualBlockWith1x1_4(x)
        x = self.up3(x)  # (B,64,H/4,W/4)
        x += x3      # Skip connection

        x = self.conv3(x)
        x = self.ResidualBlockWith1x1_3(x)
        x = self.up2(x)  # (B,32,H/2,W/2)
        x += x2      # Skip connection

        x = self.conv2(x)
        x = self.ResidualBlockWith1x1_2(x)
        x = self.up1(x)  # (B,16,H,W)
        x += x1      # Skip connection

        x = self.conv1(x)
        x = self.ResidualBlockWith1x1_1(x)

        out = self.final_conv(x)  # (B,num_classes,H,W)

        return out

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.identity_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel attention and spatial attention sequentially.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)  (same as input)
    """
    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        # --- Channel attention ---
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        # --- Spatial attention ---
        padding = (sa_kernel - 1) // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=sa_kernel,
                                 padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        ca  = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x   = x * ca

        # Spatial attention
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        sa  = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        x   = x * sa
        return x

class HaarDownsample2x(nn.Module):
    """
    Haar wavelet downsampling with channel increase ×2 (not ×4).

    Strategy:
      - Compute the 4 subbands (LL, LH, HL, HH) per input channel
      - Concatenate them along channel dim → 4C
      - Use 1×1 conv to reduce from 4C → 2C

    Input : (B, C, H, W)   with H,W even
    Output: (B, 2C, H/2, W/2)
    """
    def __init__(self, in_channels: int, normalize: bool = True):
        super().__init__()
        s = (1 / 2**0.5) if normalize else 1.0

        # Haar basis
        low  = torch.tensor([1.,  1.]) * s
        high = torch.tensor([1., -1.]) * s
        LL = torch.outer(low,  low)
        LH = torch.outer(low,  high)
        HL = torch.outer(high, low)
        HH = torch.outer(high, high)
        k = torch.stack([LL, LH, HL, HH], dim=0)  # (4,2,2)

        self.register_buffer("kernels", k.view(4,1,2,2))  # (4,1,2,2)

        # 1×1 conv to compress 4C → 2C
        self.reduce = nn.Conv2d(4*in_channels, 2*in_channels,
                                kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        w = self.kernels.expand(-1, c, -1, -1)           # (4,C,2,2)
        w = w.permute(1,0,2,3).reshape(4*c,1,2,2)        # (4C,1,2,2)
        y = F.conv2d(x, w, stride=2, groups=c)           # (B,4C,H/2,W/2)
        return self.reduce(y)                            # (B,2C,H/2,W/2)


class JointDecoderAttention(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()

        self.Conv3x3Middle = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.Conv3x3Out = nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, input: torch.Tensor, high: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:

        out = input + high

        out = torch.cat((temporal, out), dim=1)

        out = self.Conv3x3Middle(out)
        out = self.Conv3x3Out(out)

        return out


class SegDecoderJoint(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.ResidualBlockWith1x1_1 = ResidualBlockWith1x1(16,16)
        self.ResidualBlockWith1x1_2 = ResidualBlockWith1x1(32,32)
        self.ResidualBlockWith1x1_3 = ResidualBlockWith1x1(64,64)
        self.ResidualBlockWith1x1_4 = ResidualBlockWith1x1(128,128)
        self.ResidualBlockWith1x1_5 = ResidualBlockWith1x1(256,256)

        self.TempAttention1 = JointDecoderAttention(768,256)
        self.TempAttention2 = JointDecoderAttention(384,128)
        self.TempAttention3 = JointDecoderAttention(128, 64)


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

        # Final 1x1 conv to get desired number of classes
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features, temporal_features):
        x1, x2, x3, x4, x5 = features  # x1: (B,32,H/2,W/2), x2: (B,64,H/4,W/4), x3: (B,128,H/8,W/8), x4: (B,256,H/16,W/16)
        t1, t2, t3,  = temporal_features


        x = self.ResidualBlockWith1x1_5(x5)  # (B,128,H/8,W/8)
        x = self.TempAttention1(x, x5, t1)

        x = self.up4(x)# Skip connectio
        x = self.conv4(x)
        x = self.ResidualBlockWith1x1_4(x)
        x = self.TempAttention2(x, x4, t2)

        x = self.up3(x)  # (B,64,H/4,W/4)
        x += x3      # Skip connection
        x = self.conv3(x)
        x = self.ResidualBlockWith1x1_3(x)
        x = self.TempAttention3(x, x3, t3)

        x = self.up2(x)  # (B,32,H/2,W/2)
        x += x2      # Skip connection
        x = self.conv2(x)
        x = self.ResidualBlockWith1x1_2(x)

        x = self.up1(x)  # (B,16,H,W)
        x += x1      # Skip connection
        x = self.conv1(x)
        x = self.ResidualBlockWith1x1_1(x)

        out = self.final_conv(x)  # (B,num_classes,H,W)

        return out