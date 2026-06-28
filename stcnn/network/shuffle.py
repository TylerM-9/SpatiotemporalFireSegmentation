import torch
import torch.nn as nn

try:
    from torchinfo import summary
except ImportError:
    summary = None


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        if stride == 1:
            inp_main = inp // 2
        else:
            inp_main = inp

        outputs = oup - inp_main
        self.outputs = outputs

        branch_main = [
            # pw
            nn.Conv2d(inp_main, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

class PretrainedShuffleEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load full pretrained ShuffleNetV2
        base_model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
        
        # Take parts separately
        self.conv1 = base_model.conv1
        self.maxpool = base_model.maxpool
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base_model.stage4
        self.conv5 = base_model.conv5

    def forward(self, x, return_feature_maps=False):
        x = self.conv1(x)
        x = self.maxpool(x)

        feat2 = self.stage2(x)  # (B, 116, H/4, W/4)
        feat3 = self.stage3(feat2)  # (B, 232, H/8, W/8)
        feat4 = self.stage4(feat3)  # (B, 464, H/16, W/16)

        x = self.conv5(feat4)  # (B, 1024, H/16, W/16)

        if return_feature_maps:
            return [feat2, feat3, feat4, x]
        return x

if __name__ == "__main__":

    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
    model.eval()
    model.fc = torch.nn.Identity()
    summary(model, input_size=(8, 3, 300, 300))
