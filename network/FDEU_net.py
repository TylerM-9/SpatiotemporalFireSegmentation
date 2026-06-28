import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class STCNN_FDE(nn.Module):
	def __init__(self, pred_enc, pred_dec, seg):
		super(STCNN_FDE, self).__init__()
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

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and a 1x1 skip connection.

    Architecture:
        x → [3x3 Conv → BN → ReLU → 3x3 Conv → BN] → (+) → ReLU → out
            ↓                                          ↑
            └────────────→ [1x1 Conv] ────────────────┘
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # Main path (left side of diagram)
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection (right side of diagram)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Final activation after addition
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Main path
        identity = self.main_path(x)

        # Skip connection
        skip = self.skip_connection(x)

        # Element-wise addition
        out = identity + skip

        # Final ReLU
        out = self.relu(out)

        return out

class FDEUnet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(FDEUnet, self).__init__()

        features = init_features

        # Encoder (downsampling path)
        self.encoder1 = FDEUnet._block(in_channels, features, name="enc1")
        self.down1 = Down_wt(features, features)

        self.encoder2 = FDEUnet._block(features, features * 2, name="enc2")
        self.down2 = Down_wt(features * 2, features * 2)

        self.encoder3 = FDEUnet._block(features * 2, features * 4, name="enc3")
        self.down3 = Down_wt(features * 4, features * 4)

        self.encoder4 = FDEUnet._block(features * 4, features * 8, name="enc4")
        self.down4 = Down_wt(features * 8, features * 8)


        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=features * 8,
                out_channels=features * 16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features * 16,
                out_channels=features * 16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
            ResidualBlock(features * 16, features * 16),
        )

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = FDEUnet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = FDEUnet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = FDEUnet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = FDEUnet._block(features * 2, features, name="dec1")

        # Final convolution
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # Encoder

        enc1 = self.encoder1(x)

        x = self.down1(enc1)

        enc2 = self.encoder2(x)

        x = self.down2(enc2)

        enc3 = self.encoder3(x)

        x = self.down3(enc3)
        enc4 = self.encoder4(x)

        # Bottleneck
        bottleneck = self.bottleneck(self.down4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        """
        Basic U-Net block with two convolutions, batch norm, and ReLU
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            ResidualBlock(features, features),
            ACmix(features, features),
            CBAM(features),
        )

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class JointDecoderAttention(nn.Module):
    def __init__(self, n_channels, out_channels):
        super().__init__()

        self.Conv3x3Middle = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.Conv3x3Out = nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, input: torch.Tensor, high: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        out = torch.cat([input, high], dim=1)

        temporal = self.Upsample(temporal)

        out = torch.cat((temporal, out), dim=1)

        out = self.Conv3x3Middle(out)
        out = self.Conv3x3Out(out)

        return out

class FDEUnetTemporalAttention(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(FDEUnetTemporalAttention, self).__init__()

        features = init_features

        self.TempAttention1 = JointDecoderAttention(768,256)
        self.TempAttention2 = JointDecoderAttention(384,128)
        self.TempAttention3 = JointDecoderAttention(128, 64)

        # Encoder (downsampling path)
        self.encoder1 = FDEUnet._block(in_channels, features, name="enc1")
        self.down1 = Down_wt(features, features)

        self.encoder2 = FDEUnet._block(features, features * 2, name="enc2")
        self.down2 = Down_wt(features * 2, features * 2)

        self.encoder3 = FDEUnet._block(features * 2, features * 4, name="enc3")
        self.down3 = Down_wt(features * 4, features * 4)

        self.encoder4 = FDEUnet._block(features * 4, features * 8, name="enc4")
        self.down4 = Down_wt(features * 8, features * 8)


        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=features * 8,
                out_channels=features * 16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features * 16,
                out_channels=features * 16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
            ResidualBlock(features * 16, features * 16),
        )

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = FDEUnet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = FDEUnet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = FDEUnet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = FDEUnet._block(features * 2, features, name="dec1")

        # Final convolution
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, temp_features):
        # Encoder
        t1, t2, t3 = temp_features

        enc1 = self.encoder1(x)

        x = self.down1(enc1)

        enc2 = self.encoder2(x)

        x = self.down2(enc2)

        enc3 = self.encoder3(x)

        x = self.down3(enc3)
        enc4 = self.encoder4(x)

        # Bottleneck
        bottleneck = self.bottleneck(self.down4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = self.TempAttention1(dec4, enc4, t1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.TempAttention2(dec3, enc3, t2)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.TempAttention3(dec2, enc2, t3)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        """
        Basic U-Net block with two convolutions, batch norm, and ReLU
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            ResidualBlock(features, features),
            ACmix(features, features),
            CBAM(features),
        )