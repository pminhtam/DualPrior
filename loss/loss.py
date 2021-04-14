#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
import torch.nn.functional as F
import torch.nn as nn
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
def loss_fn(out_noise_1,im_denoise_1,im_noise_1,out_noise_2,im_denoise_2,im_noise_2):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
    '''
    eps = 1e-3
    im_denoise_phi_1 = im_noise_1 - out_noise_1
    im_denoise_phi_2 = im_noise_2 - out_noise_2
    diff_denoise_1 = im_noise_1 - im_denoise_1
    loss_denoise_1 = torch.mean(torch.sqrt((diff_denoise_1 * diff_denoise_1) + (eps*eps)))

    diff_denoise_phi_1 = im_denoise_1 - im_denoise_phi_1
    loss_denoise_phi_1 = torch.mean(torch.sqrt((diff_denoise_phi_1 * diff_denoise_phi_1) + (eps*eps)))

    diff_denoise_2 = im_noise_2 - im_denoise_2
    loss_denoise_2 = torch.mean(torch.sqrt((diff_denoise_2 * diff_denoise_2) + (eps*eps)))

    diff_denoise_phi_2 = im_denoise_2 - im_denoise_phi_2
    loss_denoise_phi_2 = torch.mean(torch.sqrt((diff_denoise_phi_2 * diff_denoise_phi_2) + (eps*eps)))

    # diff_noise_noise = out_noise_1 - out_noise_2
    # loss_noise_noise = torch.mean(torch.sqrt((diff_noise_noise * diff_noise_noise) + (eps*eps)))
    # loss = loss_denoise_1 + loss_denoise_2 + loss_noise_noise
    loss = loss_denoise_1 + loss_denoise_2 + loss_denoise_phi_1 + loss_denoise_phi_2

    return loss

