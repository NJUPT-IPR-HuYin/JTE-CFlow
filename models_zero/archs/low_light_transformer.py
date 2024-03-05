import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models_zero.archs.arch_util as arch_util
import numpy as np
import cv2

from models_zero.archs.transformer.Models import Encoder_patch66
from models.modules.ConditionEncoder import RRDB

from time import time


class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=4, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        # ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_f = functools.partial(RRDB, nf=nf, gc=32)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        # self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # self.cat_last = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 128, 3, 2, 1), nn.Conv2d(128, 192, 3, 2, 1), nn.Sigmoid())

        self.transformer_dual = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)

    def forward(self, x, weight, mask=None):
        x_center = x
        block_results = {}
        block_idxs = {5}

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        fea = L1_fea_2
        fea_head = fea

        for idx, m in enumerate(self.feature_extraction.children()):
            fea = m(fea)
            block_results["block_{}".format(idx)] = fea
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        trunk = trunk + fea_head

        # fea_tr = F.max_pool2d(trunk, 2)
        fea_tr = trunk
        height = fea_tr.shape[2]
        width = fea_tr.shape[3]

        weight = weight.repeat(1, 64, 1, 1)
        weight = F.interpolate(weight, size=[height, width], mode='bilinear')

        fea_unfold = F.unfold(fea_tr, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        weight_unfold = F.unfold(weight, kernel_size=4, dilation=1, stride=4, padding=0)
        weight_unfold = weight_unfold.permute(0, 2, 1)

        fea_tr = self.transformer_dual(fea_unfold, weight_unfold, src_mask=None)
        fea_tr = fea_tr.permute(0, 2, 1)
        fea_tr = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_tr)

        fea_tr = fea_tr + trunk

        out_noise = fea_tr + fea_head
        fea_down2 = out_noise
        fea_down4 = self.downconv1(
            F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)
        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))

        out_noise = self.fine_tune_color_map(fea_down2)

        results = {
                   'color_map': out_noise,
                   'fea_up0': fea_down8,
                   'fea_up1': fea_down4,
                   'fea_up2': fea_down2,
                   }
        for k, v in block_results.items():
            results[k] = v

        return results


if __name__ == '__main__':
    # print(torch.__version__)
    x = torch.randn(1, 3, 400, 600)
    y = torch.randn(1, 1, 400, 600)
    # z = torch.randn(1, 1, 400, 600)
    model = low_light_transformer(nf=64, nframes=5, groups=8, front_RBs=3, back_RBs=1, center=None,
                                  predeblur=True, HR_in=True, w_TSA=True)
    print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
    begin = time()
    z = model(x, y)
    end = time()
    print(end - begin)
    # print(z)
