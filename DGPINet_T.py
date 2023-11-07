import torch.nn as nn
import torch
import torch.nn.functional as F
import collections
from collections import OrderedDict
from model.toolbox.models.A1project2.Segfomer_Net.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
# import math
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)

class perviseHead(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(perviseHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, n_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)

class RWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(RWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim // reduction, self.dim),
                    nn.Sigmoid())

    def forward(self, r1):
        B, _, H, W = r1.shape
        avg = self.avg_pool(r1).view(B, self.dim)
        y = self.mlp(avg).view(B, self.dim, 1)
        r_weights = y.reshape(B, 1, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 1 B C 1 1
        return r_weights[0]

class DWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(DWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, d1):
        B, _, H, W = d1.shape
        # x = torch.cat((r1, d1), dim=1) # B 2C H W
        d_weights = self.mlp(d1).reshape(B, 1, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return d_weights[0]


class IAM(nn.Module):
    def __init__(self, dim, reduction=1, alpha_a=.8, beta_b=.2):
        super(IAM, self).__init__()
        self.alpha_a = alpha_a
        self.beta_b = beta_b
        self.R_weights = RWeights(dim=dim, reduction=reduction)
        self.D_weights = DWeights(dim=dim, reduction=reduction)

    def forward(self, r, d):
        r_w1 = self.R_weights(r)
        d_w1 = self.D_weights(d)
        out_r = r + self.alpha_a * r_w1 * r + self.beta_b * d_w1 * d

        return out_r

class depth_k(nn.Module):
    def __init__(self, channel4=512):
        super(depth_k, self).__init__()

        self.fuse4 = convbnrelu(channel4, channel4, k=1, s=1, p=0, relu=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth4 = DSConv3x3(channel4, channel4, stride=1, dilation=1)

    def forward(self, x4, k4):
        B4, C4, H4, W4 = k4.size()
        x_B4, x_C4, x_H4, x_W4 = x4.size()
        x4_new = x4.clone()

        for i in range(1, B4):
            kernel4 = k4[i, :, :, :]
            kernel4 = kernel4.view(C4, 1, H4, W4)
            x4_r1 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=2, dilation=1,
                             groups=C4)
            x4_r2 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=4, dilation=2,
                             groups=C4)
            x4_r3 = F.conv2d(x4[i, :, :, :].view(1, C4, x_H4, x_W4), kernel4, stride=1, padding=6, dilation=3,
                             groups=C4)
            x4_new[i, :, :, :] = x4_r1 + x4_r2 + x4_r3

        x4_all = self.fuse4(x4_new)
        x4_smooth = self.smooth4(x4_all)
        return x4_smooth
class RaD_fusion(nn.Module):
    def __init__(self, channel):
        super(RaD_fusion, self).__init__()
        self.maskr_1 = nn.Conv2d(channel, 1, 1)
        self.maskd_1 = nn.Conv2d(channel, 1, 1)
        self.mask_rd = convbnrelu(2, 2)
    def forward(self, r, d):
        maskr1 = self.maskr_1(r)
        maskd1 = self.maskd_1(d)
        maskrd1 = torch.cat((maskr1, maskd1), dim=1)
        split_maskrd = self.mask_rd(maskrd1)
        r1, d1 = split_maskrd.split(1, dim=1)
        ro = r * r1
        do = d * d1
        gout = ro + do
        return gout

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_wish(nn.Module):
    def __init__(self, inplace=True):
        super(h_wish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class Ca_block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(Ca_block, self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.con1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
        self.act = h_wish()
        self.bn = nn.BatchNorm2d(channel//reduction)
        self.h_con = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.w_con = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self, input):
        input_h = self.avg_pool_x(input).permute(0, 1, 3, 2)
        input_w = self.avg_pool_y(input)
        input_ccr = self.act(self.con1(torch.cat((input_h, input_w), 3)))
        input_split_h, input_split_w = input_ccr.split([self.h, self.w], 3)
        sig_h = self.sigmoid_h(self.h_con(input_split_h.permute(0, 1, 3, 2)))
        sig_w = self.sigmoid_w(self.w_con(input_split_w))
        out = input * sig_h.expand_as(input) * sig_w.expand_as(input)
        return out
    
class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):

        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        out = self.relu(buffer_1+self.conv_res(x))

        return out
class fusion12(nn.Module):
    def __init__(self, inc, h, w):
        super(fusion12, self).__init__()
        self.r_ca = Ca_block(channel=inc, h=h, w=w)
        self.sof = nn.Softmax(dim=1)
        self.er = ER(in_channel=inc)
    def forward(self, r, d):
        rd = r + d
        out = self.r_ca(rd)
        out = self.er(out)
        return out

class pp_upsample(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
class EnDecoderModel(nn.Module):
    def __init__(self, n_classes, backbone):
        super(EnDecoderModel, self).__init__()
        if backbone == 'segb4':
            self.backboner = mit_b4()
            self.backboned = mit_b4()

        # 1-2_levels_fusion
        self.fusions = nn.ModuleList([
            fusion12(64, 120, 160),
            fusion12(128, 60, 80),
        ])
        # 3-4_levels_K
        self.f4_k4 = DSConv3x3(512, 512, stride=1)   # k4:512
        self.f4_k3 = DSConv3x3(320, 320, stride=1)   # k3:320
        self.pool = nn.AdaptiveAvgPool2d(5)          # k4:512*5*5, k3:320*5*5
        self.depth_k3 = depth_k(320)
        self.depth_k4 = depth_k(512)
        self.IAM4 = IAM(dim=512, reduction=1)
        self.IAM3 = IAM(dim=320, reduction=1)
        #############################################
        channels = [64, 128, 320, 512]
        self.local_att3 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )
        self.global_att3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att4 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )
        self.global_att4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(80, 320, 1, 1)
        self.conv4 = nn.Conv2d(128, 512, 1, 1)
        #############################################
        self.con4 = convbnrelu(512, 320, k=1, s=1, p=0)
        self.con3 = convbnrelu(320, 128, k=1, s=1, p=0)
        self.con2 = convbnrelu(128, 64, 1, 1)
        self.c11 = convbnrelu(64, n_classes, 1, 1)

        self.rd_fusion4 = RaD_fusion(512)
        self.rd_fusion3 = RaD_fusion(320)

        self.rd_fusion34 = RaD_fusion(320)
        self.rd_fusion234 = RaD_fusion(128)

        self.f4_p = perviseHead(320, n_classes)
        self.f3_p = perviseHead(128, n_classes)
        self.f2_p = perviseHead(64, n_classes)

        self.Decon_out1 = pp_upsample(64, 64)
        self.Decon_out2 = pp_upsample(64, 64)
        self.Decon_out320 = pp_upsample(320, 320)
        self.Decon_out128 = pp_upsample(128, 128)
        self.Decon_out64 = pp_upsample(64, 64)
    def forward(self, rgb, dep):

        features_rgb = self.backboner(rgb)
        features_dep = self.backboned(dep)
        features_rlist = features_rgb[0]
        features_dlist = features_dep[0]

        rf1 = features_rlist[0]
        rf2 = features_rlist[1]
        rf3 = features_rlist[2]
        rf4 = features_rlist[3]

        df1 = features_dlist[0]
        df2 = features_dlist[1]
        df3 = features_dlist[2]
        df4 = features_dlist[3]
        FD_pervise = []
        FD1 = self.fusions[0](rf1, df1)
        FD2 = self.fusions[1](rf2, df2)

        #############################################
        fd4 = rf4 + df4
        fd4l = self.local_att4(fd4)
        fd4g = self.global_att4(fd4)
        w4 = self.sigmoid(self.conv4(fd4l+fd4g)+fd4)
        FD44 = rf4 * w4 + df4 * (1 - w4)
        F4 = rf4 * w4
        iam_r4, iam_d4 = self.IAM4(F4, df4)
        kernel_4d = self.pool(self.f4_k4(df4))  # K4d:512*5*5
        FD4 = self.depth_k4(iam_r4, kernel_4d) + FD44
        ##############################################
        fd3 = rf3 + df3
        fd3l = self.local_att3(fd3)
        fd3g = self.global_att3(fd3)
        w3 = self.sigmoid(self.conv3(fd3l+fd3g)+fd3)
        FD33 = rf3 * w3 + df3 * (1 - w3)
        F3 = rf3 * w3
        iam_r3, iam_d3 = self.IAM3(F3, df3)
        kernel_3d = self.pool(self.f4_k3(df3))  # K3d:320*5*5
        FD3 = self.depth_k3(iam_r3, kernel_3d) + FD33
        #############################################
        #############################################
        FD4 = self.con4(FD4)
        FD4_p = self.f4_p(FD4)

        FD_pervise.append(FD4_p)        # FD_p[0]_c:320->41
        FD4_2 = self.Decon_out320(FD4)

        FD34 = self.rd_fusion34(FD3, FD4_2)

        FD34 = self.con3(FD34)
        FD3_p = self.f3_p(FD34)
        FD_pervise.append(FD3_p)        # FD_p[1]_c:128->41
        FD34_2 = self.Decon_out128(FD34)

        FD234 = self.rd_fusion234(FD2, FD34_2)
        FD234 = self.con2(FD234)
        FD2_p = self.f2_p(FD234)
        FD_pervise.append(FD2_p)        # FD_p[2]_c:64->41
        FD234_2 = self.Decon_out64(FD234)


        out = FD1 + FD234_2
        out = self.Decon_out1(out)
        out = self.Decon_out2(out)
        out = self.c11(out)

        return out, FD_pervise, features_rlist

    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backboner.load_state_dict(new_state_dict3, strict=False)
        self.backboned.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone loading')

if __name__ == '__main__':
     import os
     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     device = torch.device('cuda')
     rgb = torch.randn(2, 3, 480, 640).to(device)
     dep = torch.randn(2, 3, 480, 640).to(device)
     model = EnDecoderModel(n_classes=38, backbone='segb4').to(device)
     out = model(rgb, dep)
     print('out[1]输出结果：', out[0].shape)
     for i in out[1]:
        print('out[1]输出结果：', i.shape)
