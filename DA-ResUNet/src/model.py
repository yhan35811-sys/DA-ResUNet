import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


# Part 1: 基础组件


class DoubleConv(nn.Module):
    

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



# Part 2: 注意力模块


class ChannelAttention(nn.Module):
    

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_v = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + max_v)


class SpatialAttention(nn.Module):


    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class DualAttentionBlock(nn.Module):
    

    def __init__(self, in_channels, mode='dual'):
        super(DualAttentionBlock, self).__init__()
        self.mode = mode
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        if self.mode == 'none':
            return x
        elif self.mode == 'channel':
            return x * self.ca(x)
        elif self.mode == 'spatial':
            return x * self.sa(x)
        else:  # dual
            return x * self.ca(x) * self.sa(x)



# Part 3:  Attention U-Net


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Classic_AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Classic_AttentionUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = DoubleConv(img_ch, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.Att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(512, 256)

        self.Up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, 1)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1);
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2);
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3);
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4);
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)



# Part 4:  DA-ResUNet


class DA_ResUNet(nn.Module):
    def __init__(self, n_classes=1, attention_mode='dual'):
        super().__init__()
        # Encoder: ResNet34 (Updated weights)
        base = models.resnet34(weights='DEFAULT')
        self.base_layers = list(base.children())

        self.enc0 = nn.Sequential(*self.base_layers[:3])  # 64
        self.enc1 = nn.Sequential(*self.base_layers[4])  # 64
        self.enc2 = nn.Sequential(*self.base_layers[5])  # 128
        self.enc3 = nn.Sequential(*self.base_layers[6])  # 256
        self.enc4 = nn.Sequential(*self.base_layers[7])  # 512

        # Decoder with Dual Attention
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = DualAttentionBlock(512, mode=attention_mode)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = DualAttentionBlock(256, mode=attention_mode)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = DualAttentionBlock(128, mode=attention_mode)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.att1 = DualAttentionBlock(128, mode=attention_mode)
        self.dec1 = DoubleConv(128, 64)

        #  128x128 -> 256x256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec_final = DoubleConv(32, 32)
        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x);
        e1 = self.enc1(e0);
        e2 = self.enc2(e1);
        e3 = self.enc3(e2);
        e4 = self.enc4(e3)

        d4 = self.up4(e4)
        if d4.shape != e3.shape: d4 = F.interpolate(d4, size=e3.shape[2:])
        d4 = self.dec4(self.att4(torch.cat([e3, d4], 1)))

        d3 = self.up3(d4)
        if d3.shape != e2.shape: d3 = F.interpolate(d3, size=e2.shape[2:])
        d3 = self.dec3(self.att3(torch.cat([e2, d3], 1)))

        d2 = self.up2(d3)
        if d2.shape != e1.shape: d2 = F.interpolate(d2, size=e1.shape[2:])
        d2 = self.dec2(self.att2(torch.cat([e1, d2], 1)))

        d1 = self.up1(d2)
        if d1.shape != e0.shape: d1 = F.interpolate(d1, size=e0.shape[2:])
        d1 = self.dec1(self.att1(torch.cat([e0, d1], 1)))

        d0 = self.up_final(d1)
        d0 = self.dec_final(d0)

        return self.final(d0)



# Part 5: 模型选择，根据要求切换模型；


def get_model():
    n_classes = 1


   
    model = DA_ResUNet(n_classes=n_classes, attention_mode='dual')


    

    # 2.  Attention U-Net 
    #model = Classic_AttentionUNet(img_ch=3, output_ch=n_classes)

   

    return model