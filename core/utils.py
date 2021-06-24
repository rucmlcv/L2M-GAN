"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import random


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref, s_ref1, s_ref2, s_ref3 = nets.style_encoder(x_ref, y_src, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref1 + s_ref3, masks=masks)
    s_src, _, _, _ = nets.style_encoder(x_src, y_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat

@torch.no_grad()
def translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)
    x_ref_with_wb = torch.cat([wb, x_ref], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_concat = [x_src_with_wb]
    x_concat += [x_ref_with_wb]
    for i in range(args.num_domains) : 
        s_src, s_src1, s_src2, s_src3 = nets.style_encoder(x_src, y_src, y_ref)
        s_ref, s_ref1, s_ref2, s_ref3 = nets.style_encoder(x_ref, y_ref, y_src)
        x_fake = nets.generator(x_src, s_src1 + s_ref2, masks=masks)
        x_fake_with_ref = torch.cat([wb, x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    print(x_concat.shape)
    save_image(x_concat, N+1, filename)
    del x_concat

@torch.no_grad()
def translate_self(nets, args, x_src, y_src, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_concat = [x_src_with_wb]
    for i in range(args.num_domains) : 
        y_trg = np.zeros(x_src.size(0))
        for j in range(x_src.size(0)) : 
            y_trg[j] = i
        y_ref = torch.from_numpy(y_trg).long().to(x_src.device)
        s_ref, s_ref1, s_ref2, s_ref3 = nets.style_encoder(x_src, y_src, y_ref)
        #s_ref = nets.style_transform(s_ref)
        x_fake = nets.generator(x_src, s_ref1 + s_ref3, masks=masks)
        x_fake1 = nets.generator(x_src, s_ref1 + (s_ref2 + 1.25 * (s_ref3 - s_ref2)), masks=masks)
        x_fake_with_ref = torch.cat([wb, x_fake], dim=0)
        x_fake_with_ref1 = torch.cat([wb, x_fake1], dim=0)
        x_concat += [x_fake_with_ref]
        x_concat += [x_fake_with_ref1]
        for j in range(N) : 
            save_image(x_fake[j], 1, ospj(args.result_dir, '%02d.png' % (j)))
            save_image(x_fake1[j], 1, ospj(args.result_dir, '%02d_1.png' % (j)))
    
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src

    device = inputs.x_src.device
    N = inputs.x_src.size(0)
    y_ref = np.zeros(N)
    for i in range(N) :
        while y_ref[i] == y_src[i] :
            y_ref[i] = random.randint(0,args.num_domains-1)
    y_ref = torch.from_numpy(y_ref).long().to(device)
    
    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_src, y_ref, filename)