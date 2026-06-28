import torch
import torch.nn as nn
import torch.nn.functional as F

class STCNNRES(nn.Module):
	def __init__(self, pred_enc, pred_dec, seg):
		super(STCNNRES, self).__init__()
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
        out = torch.cat([input, high], dim=1)

        temporal = self.Upsample(temporal)

        out = torch.cat((temporal, out), dim=1)

        out = self.Conv3x3Middle(out)
        out = self.Conv3x3Out(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += self.skip_connection(residual)
        out = self.relu(out)

        return out

class ResNetJointDecoderAttention(nn.Module):
    """ResNet with Joint Decoder Attention"""

    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(ResNetJointDecoderAttention, self).__init__()

        features = init_features

        self.TempAttention1 = JointDecoderAttention(768,256)
        self.TempAttention2 = JointDecoderAttention(384,128)
        self.TempAttention3 = JointDecoderAttention(128, 64)

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, features, kernel_size=3,
                                      padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

        # Encoder (Contracting Path)
        self.encoder1 = ResidualBlock(features, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ResidualBlock(features, features * 2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ResidualBlock(features * 2, features * 4, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ResidualBlock(features * 4, features * 8, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(features * 8, features * 16, stride=1)

        # Decoder (Expansive Path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8,
                                          kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(features * 16, features * 8, stride=1)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4,
                                          kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features * 8, features * 4, stride=1)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2,
                                          kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features * 4, features * 2, stride=1)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features,
                                          kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features * 2, features, stride=1)

        # Output layer
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x, temp_features):
        t1, t2, t3 = temp_features

        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)

        enc3 = self.encoder3(x)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv4(x)
        x = self.TempAttention1(x, enc4, t1)  # Joint Decoder Attention
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = self.TempAttention2(x, enc3, t2)  # Joint Decoder Attention
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = self.TempAttention3(x, enc2, t3)  # Joint Decoder Attention
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.decoder1(x)

        # Output
        x = self.final_conv(x)

        return x

        return x

class ResUNet(nn.Module):
    """ResUNet: U-Net with Residual Blocks"""

    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(ResUNet, self).__init__()

        features = init_features

        self.TempAttention1 = JointDecoderAttention(768,256)
        self.TempAttention2 = JointDecoderAttention(384,128)
        self.TempAttention3 = JointDecoderAttention(128, 64)

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, features, kernel_size=3,
                                      padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

        # Encoder (Contracting Path)
        self.encoder1 = ResidualBlock(features, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ResidualBlock(features, features * 2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ResidualBlock(features * 2, features * 4, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ResidualBlock(features * 4, features * 8, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(features * 8, features * 16, stride=1)

        # Decoder (Expansive Path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8,
                                          kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(features * 16, features * 8, stride=1)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4,
                                          kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features * 8, features * 4, stride=1)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2,
                                          kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features * 4, features * 2, stride=1)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features,
                                          kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features * 2, features, stride=1)

        # Output layer
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)

        enc3 = self.encoder3(x)
        x = self.pool3(enc3)

        enc4 = self.encoder4(x)
        x = self.pool4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)  # Skip connection
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.decoder1(x)

        # Output
        x = self.final_conv(x)

        return x

