# --- NEW: attention modules ---

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Basic double conv block ----
class ConvBlock(nn.Module):
    """
    Two Conv-BN-ReLU layers (classic U-Net block).
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ---- Basic UpBlock (used for the final stage, without attention) ----
class UpBlock(nn.Module):
    """
    Upsample + concatenate skip + ConvBlock.
    Used for the last stage (no temporal/attention).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            conv_in = out_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
            conv_in = out_ch + skip_ch
        self.block = ConvBlock(conv_in, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        # pad if needed (for odd shapes)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ---- Encoder (unchanged from your previous version) ----
class UNetEncoder(nn.Module):
    """
    Encoder producing multi-scale features.
    Returns: (e1, e2, e3, e4), bottleneck
      e1: 1/1,  e2: 1/2,  e3: 1/4,  e4: 1/8, bottleneck: 1/16
    """
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)       # 1/1
        self.p1 = nn.MaxPool2d(2)              # 1/2

        self.e2 = ConvBlock(base, base * 2)    # 1/2
        self.p2 = nn.MaxPool2d(2)              # 1/4

        self.e3 = ConvBlock(base * 2, base * 4) # 1/4
        self.p3 = nn.MaxPool2d(2)              # 1/8

        self.e4 = ConvBlock(base * 4, base * 8) # 1/8
        self.p4 = nn.MaxPool2d(2)              # 1/16

        self.bottleneck = ConvBlock(base * 8, base * 16)

    def forward(self, x: torch.Tensor):
        e1 = self.e1(x)
        x = self.p1(e1)

        e2 = self.e2(x)
        x = self.p2(e2)

        e3 = self.e3(x)
        x = self.p3(e3)

        e4 = self.e4(x)
        x = self.p4(e4)

        b = self.bottleneck(x)
        return (e1, e2, e3, e4), b

class AttentionFusion(nn.Module):
    """
    STCNN-style fusion:
      1) element-wise add high-level context (after 1x1 proj),
      2) concat temporal feature (if given),
      3) gate with upsampled previous mask (if given),
      4) refine via two 3x3 convs.
    """
    def __init__(self, in_spatial: int, in_temporal: int, out_ch: int):
        super().__init__()
        self.has_temporal = in_temporal > 0
        self.proj_s = nn.Conv2d(in_spatial, out_ch, kernel_size=1, bias=False)
        if self.has_temporal:
            self.proj_t = nn.Conv2d(in_temporal, out_ch, kernel_size=1, bias=False)
            fuse_in = out_ch * 2  # add -> (s+t), then concat t  => [out,out] -> 2*out
        else:
            self.proj_t = None
            fuse_in = out_ch

        self.refine = nn.Sequential(
            nn.Conv2d(fuse_in, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial: torch.Tensor,
                temporal: torch.Tensor = None,
                prev_mask: torch.Tensor = None) -> torch.Tensor:
        s = self.proj_s(spatial)
        if self.has_temporal and (temporal is not None):
            t = self.proj_t(temporal)
            x_add = s + t                     # (1) add context
            x = torch.cat([x_add, t], dim=1)  # (2) concat temporal cue
        else:
            x = s

        if prev_mask is not None:
            # (3) mask-guided gating (resize to current feature size)
            pm = F.interpolate(prev_mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x * pm.clamp(0, 1)  # assumes mask in [0,1]

        # (4) refine
        return self.refine(x)


class AttentionUpBlock(nn.Module):
    """
    Upsample -> concat skip -> AttentionFusion -> (optional) aux head
    Accepts optional temporal feature at this scale and previous-stage mask.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 temporal_ch: int = 0, bilinear: bool = True, aux_pred: bool = False):
        super().__init__()
        self.bilinear = bilinear
        self.aux_pred = aux_pred

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            fuse_spatial_in = out_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
            fuse_spatial_in = out_ch + skip_ch

        # 1x1 to pre-mix spatial before attention (keeps semantics similar to classic UpBlock)
        self.pre = nn.Sequential(
            nn.Conv2d(fuse_spatial_in, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.fuse = AttentionFusion(in_spatial=out_ch, in_temporal=temporal_ch, out_ch=out_ch)

        if self.aux_pred:
            self.head = nn.Conv2d(out_ch, 1, kernel_size=1)  # binary aux head

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                temporal: torch.Tensor = None,
                prev_mask: torch.Tensor = None):
        x = self.up(x)
        x = self.reduce(x)

        # pad in case of odd shapes
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.pre(x)
        x = self.fuse(x, temporal=temporal, prev_mask=prev_mask)

        aux = None
        if self.aux_pred:
            aux = torch.sigmoid(self.head(x))
        return x, aux

# ---- REPLACED: Decoder with attention & multi-scale preds ----
class UNetDecoder(nn.Module):
    """
    Decoder with STCNN-style attention at 1/8, 1/4, 1/2 scales.
    Optionally returns aux predictions for deep supervision.
    Temporal features are optional: pass a dict with keys {'1/8','1/4','1/2'}.
    """
    def __init__(self, base: int = 64, out_ch: int = 1,
                 bilinear: bool = True, final_act: str = "sigmoid",
                 use_aux: bool = True, temporal_channels: dict = None):
        super().__init__()
        temporal_channels = temporal_channels or {}
        t8  = temporal_channels.get('1/8', 0)
        t4  = temporal_channels.get('1/4', 0)
        t2  = temporal_channels.get('1/2', 0)

        # attention up blocks with per-scale temporal in-ch
        self.up4 = AttentionUpBlock(base * 16, base * 8,  base * 8,  temporal_ch=t8, bilinear=bilinear, aux_pred=use_aux)  # -> 1/8
        self.up3 = AttentionUpBlock(base * 8,  base * 4,  base * 4,  temporal_ch=t4, bilinear=bilinear, aux_pred=use_aux)  # -> 1/4
        self.up2 = AttentionUpBlock(base * 4,  base * 2,  base * 2,  temporal_ch=t2, bilinear=bilinear, aux_pred=use_aux)  # -> 1/2
        self.up1 = UpBlock        (base * 2,  base,      base,      bilinear)  # final stage (spatial only)

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)
        self.final_act = final_act
        self.use_aux = use_aux

    def forward(self,
                skips_bottleneck: Tuple[Tuple[torch.Tensor, ...], torch.Tensor],
                temporal_feats: dict = None):
        """
        temporal_feats (optional): {
          '1/8': Tensor [N, Ct8, H/8,  W/8],
          '1/4': Tensor [N, Ct4, H/4,  W/4],
          '1/2': Tensor [N, Ct2, H/2,  W/2],
        }
        Returns:
          - y: final prediction
          - aux dict (if use_aux): {'1/8': m8, '1/4': m4, '1/2': m2}
        """
        temporal_feats = temporal_feats or {}
        (e1, e2, e3, e4), b = skips_bottleneck

        # stage @ 1/8 (coarsest attention); no prev mask yet
        x, m8 = self.up4(b, e4, temporal=temporal_feats.get('1/8'), prev_mask=None)

        # stage @ 1/4 (use previous aux mask to guide)
        x, m4 = self.up3(x, e3, temporal=temporal_feats.get('1/4'), prev_mask=m8)

        # stage @ 1/2 (use previous aux mask)
        x, m2 = self.up2(x, e2, temporal=temporal_feats.get('1/2'), prev_mask=m4)

        # final spatial up to 1/1 (no attention)
        x = self.up1(x, e1)

        y = self.head(x)
        if self.final_act == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_act == "softmax":
            y = F.softmax(y, dim=1)

        if self.use_aux:
            aux = {'1/8': m8, '1/4': m4, '1/2': m2}
            return y, aux
        return y

# ---- Full model wrapper (unchanged API; extras are optional) ----
class UNet(nn.Module):
    """
    If you have temporal features, pass them in forward() as a dict.
    Otherwise just call forward(x) as before.
    """
    def __init__(self, in_ch: int = 3, base: int = 64, out_ch: int = 1,
                 bilinear: bool = True, final_act: str = "sigmoid", use_aux: bool = True,
                 temporal_channels: dict = None):
        super().__init__()
        self.encoder = UNetEncoder(in_ch=in_ch, base=base)
        self.decoder = UNetDecoder(base=base, out_ch=out_ch, bilinear=bilinear,
                                   final_act=final_act, use_aux=use_aux,
                                   temporal_channels=temporal_channels)

    def forward(self, x: torch.Tensor, temporal_feats: dict = None):
        skips, b = self.encoder(x)
        return self.decoder((skips, b), temporal_feats=temporal_feats)
