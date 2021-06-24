import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips

import os
import cv2


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--space', choices=['z', 'w'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eps', type=float, default=1e-4)

    args = parser.parse_args()

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )
    
    smile_root = "/home/guoxing_yang/stargan_v2/expr/eval_8_31_ppl/sad2smile"
    sad_root = "/home/guoxing_yang/stargan_v2/expr/eval_8_31_ppl/smile2sad"
    
    smile_images_name = os.listdir(smile_root)
    sad_images_name = os.listdir(sad_root)
    smile_images_name.sort()
    sad_images_name.sort()
    
    smile_images = []
    sad_images = []
    
    for name in smile_images_name : 
        image = cv2.imread(os.path.join(smile_root, name))
        image = cv2.resize(image,(256,256))
        smile_images.append(image)
        
    for name in sad_images_name : 
        image = cv2.imread(os.path.join(sad_root, name))
        image = cv2.resize(image,(256,256))
        sad_images.append(image)
    
    
    
    n_sample = len(smile_images)
    n_batch = n_sample // args.batch
    resid = n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]
    
    n = 0
    distances = []
    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            images = smile_images[n*args.batch:n*args.batch + batch]
            images = np.array(images).transpose(0,3,1,2)
            images = torch.from_numpy(images).cuda()

            dist = percept(images[::2], images[1::2]).view(images.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to('cpu').numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    
    print('ppl:', filtered_dist.mean())
    
    
    n_sample = len(sad_images)
    n_batch = n_sample // args.batch
    resid = n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]
    
    n = 0
    distances = []
    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            images = sad_images[n*args.batch:n*args.batch + batch]
            images = np.array(images).transpose(0,3,1,2)
            images = torch.from_numpy(images).cuda()

            dist = percept(images[::2], images[1::2]).view(images.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to('cpu').numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print('ppl:', filtered_dist.mean())
