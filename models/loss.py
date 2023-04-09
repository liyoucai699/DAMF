import torch


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, loss_l = True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 1e-5
    C2 = 1e-5
    # LH
    if loss_l:
        ssim_map = 1 - (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    else:
    #Ll
        ssim_map = 1 - (sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_l=True):
        super(SSIM, self).__init__()
        self.loss_l = loss_l
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, self.loss_l)


def getsocre(img1, img2, window_size=11, size_average=True, loss_l=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, loss_l=loss_l)

def color_loss(uw_img, cl_img):
    # def __init__(self):
    #     super(color_loss, self).__init__()
    #
    # def forward(self, uw_img, cl_img):
    b, c, h, w = uw_img.shape

    mean_rgb_uw = torch.mean(uw_img, [2, 3], keepdim=True)
    mean_rgb_cl = torch.mean(cl_img, [2, 3], keepdim=True)
    uw_r, uw_g, uw_b = torch.split(mean_rgb_uw, 1, dim=1)
    cl_r, cl_g, cl_b = torch.split(mean_rgb_cl, 1, dim=1)
    d_r = torch.pow(uw_r - cl_r, 2)
    d_g = torch.pow(uw_g - cl_g, 2)
    d_b = torch.pow(uw_b - cl_b, 2)
    d = torch.pow(torch.pow(d_r, 2) * 2 + torch.pow(d_g, 2) * 5 + torch.pow(d_b, 2) * 3, 0.5)

    return d.sum().item()
