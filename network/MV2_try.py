# pip install timm
import timm
import torch
import torch.nn as nn
from typing import List, Tuple

class SegEncoder_MobileViT2_Compat(nn.Module):
    """
    MobileViT-v2 encoder that returns 4 feature maps shaped like ResNet101's:
      - conv_out = [C2, C3, C4, C5] with channels [256, 512, 1024, 2048]
    This lets SegDecoderCBAM() run unchanged.
    """
    def __init__(self,
                 mv2_variant: str = "mobilevitv2_100",
                 pretrained: bool = True,
                 out_indices: Tuple[int, ...] = (1, 2, 3, 4),
                 target_planes: Tuple[int, ...] = (256, 512, 1024, 2048)):
        super().__init__()

        # MobileViT-v2 backbone with multi-scale outputs
        self.backbone = timm.create_model(
            mv2_variant, pretrained=pretrained, features_only=True, out_indices=out_indices
        )
        fi = self.backbone.feature_info  # timm.FeatureInfo

        self.src_channels: List[int] = [fi[i]['num_chs'] for i in range(len(fi)) if i in out_indices]
        self.reductions:  List[int] = [fi[i]['reduction'] for i in range(len(fi)) if i in out_indices]
        assert len(self.src_channels) == 4, f"Expected 4 scales, got {len(self.src_channels)}"

        # 1x1 conv neck to map MobileViT channels -> ResNet-like channels expected by SegDecoderCBAM
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
            for cin, cout in zip(self.src_channels, target_planes)
        ])

        self.target_planes = target_planes  # (256, 512, 1024, 2048)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x: torch.Tensor, return_feature_maps: bool = False):
        feats = self.backbone(x)  # [C2, C3, C4, C5] MobileViT-v2 native channels
        # Map to ResNet-like channels
        feats_mapped = [neck(f) for neck, f in zip(self.neck, feats)]
        if return_feature_maps:
            return feats_mapped  # [C2, C3, C4, C5] -> channels (256, 512, 1024, 2048)
        return [feats_mapped[-1]]  # parity with your old encoder: top only when not requesting maps
