import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros, NN
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 128)
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine_1 = NN(in_channels=self.channels_for_nn,
                            out_channels=self.channels_for_co * 2,
                            hidden_channels=self.hidden_channels,
                            ft_channels=self.in_channels_rrdb)

        self.fAffine_2 = NN(in_channels=self.channels_for_nn,
                            out_channels=self.channels_for_co * 2,
                            hidden_channels=self.hidden_channels,
                            ft_channels=self.in_channels_rrdb)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = thops.split_feature(z, "cross")
            scale_1, shift_1 = self.feature_extract_aff(z2, ft, self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z1 = z1 + shift_1
            z1 = z1 * scale_1
            logdet = logdet + self.get_logdet(scale_1)
            z = thops.cat_feature(z1, z2, "norm")

            z1, z2 = thops.split_feature(z, "split")
            scale_2, shift_2 = self.feature_extract_aff(z1, ft, self.fAffine_2)
            self.asserts(scale_2, shift_2, z1, z2)
            z2 = z2 + shift_2
            z2 = z2 * scale_2
            logdet = logdet + self.get_logdet(scale_2)
            z = thops.cat_feature(z1, z2, "cross")

            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = thops.split_feature(z, "cross")
            scale_2, shift_2 = self.feature_extract_aff(z1, ft, self.fAffine_2)
            self.asserts(scale_2, shift_2, z1, z2)
            z2 = z2 / scale_2
            z2 = z2 - shift_2
            z = thops.cat_feature(z1, z2, "norm")
            logdet = logdet - self.get_logdet(scale_2)

            # Self Conditional
            z1, z2 = thops.split_feature(z, "split")
            scale_1, shift_1 = self.feature_extract_aff(z2, ft, self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z1 = z1 / scale_1
            z1 = z1 - shift_1
            z = thops.cat_feature(z1, z2, "cross")
            logdet = logdet - self.get_logdet(scale_1)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        # scale = torch.clamp(scale, 0.1, 1)
        return scale, shift

    def feature_extract_aff(self, z, ft, f):
        # z = torch.cat([z1, ft], dim=1)
        h = f(z, ft)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        # scale = torch.clamp(scale, 0.1, 1)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)
