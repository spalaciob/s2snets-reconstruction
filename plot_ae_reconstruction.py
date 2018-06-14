#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: Reconstruct image using an S2SNet AE

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.
@author: Sebastian Palacio
"""

import sys
import cv2
import torch
import argparse
import traceback
import numpy as np
from scipy.misc import imread, imresize, imsave

from models.segnet_autoencoder import SegNet


MEAN_TORCH_BGR = np.array((103.53, 116.28, 123.675), dtype=np.float32).reshape((1, 3, 1, 1))
STD_TORCH_BGR = np.array((57.375, 57.12, 58.395), dtype=np.float32).reshape((1, 3, 1, 1))
RESNET50_PATH = './s2snet_resnet50.pth'


def bgr2rgb(t, dim):
    b, g, r = torch.split(t, 1, dim)
    return torch.cat((r, g, b), dim)


def chw2hwc(im):
    return im.transpose((1, 2, 0))


def toimage_rgb(v, index=-1, dtype=np.uint8):
    v = bgr2rgb(v, 1)
    mean = np.array((103.53, 116.28, 123.675), dtype=np.float32).reshape((1, 3, 1, 1))
    std = np.array((57.375, 57.12, 58.395), dtype=np.float32).reshape((1, 3, 1, 1))
    a = v.data.cpu().numpy() * std + mean
    a = np.clip(a[index], 0, 255)
    return chw2hwc(a.astype(dtype))


def torgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def toimage(a, dtype=np.uint8):
    return torgb(toimage_rgb(a, dtype=dtype))


def make_newnet(path):
    net = SegNet(3, 3)
    state = torch.load(path)
    net.load_state_dict(state.get('model_state', state))
    return net.eval()


def main(opts):
    s2s_net = make_newnet(RESNET50_PATH).cuda()

    img = imresize(imread(opts.img), (256, 256))
    norm_img = (img[..., [2, 1, 0]] - MEAN_TORCH_BGR.flat) / STD_TORCH_BGR.flat
    norm_img = np.transpose(norm_img, (2, 0, 1))[np.newaxis,...]
    norm_img = torch.autograd.Variable(torch.Tensor(norm_img[:, [2, 1, 0], ...])).cuda()  # BGR transposition and torch-ification
    ae_img = toimage(s2s_net(norm_img), dtype=np.float32)

    _ = imsave('%s-ResNet50.png' % opts.img.split('/')[-1].split('.')[0], ae_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', metavar='FILE', required=True, help='Input image')

    opts = parser.parse_args(sys.argv[1:])

    try:
        main(opts)
    except:
        print 'Unhandled error!'
        traceback.print_exc()
        sys.exit(-1)

    print 'All Done.'
