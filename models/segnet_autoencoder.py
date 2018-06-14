from __future__ import print_function, division

import torch
from torch import nn
from torch.nn import functional as F

from . import TurboModule
from . import strip_params
from . import filter_state


class SegNetEncoder(TurboModule):
    def __init__(self, n_inputs=3, bn_momentum=0.1):
        nn.Module.__init__(self)
        self.n_inputs = n_inputs

        self.conv11 = nn.Conv2d(n_inputs, 64, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn22 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn33 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn43 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn53 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.pool53 = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        # Stage 1e
        x = F.relu(self.bn11(self.conv11(x)), inplace=True)
        x = F.relu(self.bn12(self.conv12(x)), inplace=True)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)

        # Stage 2e
        x = F.relu(self.bn21(self.conv21(x)), inplace=True)
        x = F.relu(self.bn22(self.conv22(x)), inplace=True)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)

        # Stage 3e
        x = F.relu(self.bn31(self.conv31(x)), inplace=True)
        x = F.relu(self.bn32(self.conv32(x)), inplace=True)
        x = F.relu(self.bn33(self.conv33(x)), inplace=True)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)

        # Stage 4e
        x = F.relu(self.bn41(self.conv41(x)), inplace=True)
        x = F.relu(self.bn42(self.conv42(x)), inplace=True)
        x = F.relu(self.bn43(self.conv43(x)), inplace=True)
        x, ind4 = F.max_pool2d(x, 2, 2, return_indices=True)

        # Stage 5e
        x = F.relu(self.bn51(self.conv51(x)), inplace=True)
        x = F.relu(self.bn52(self.conv52(x)), inplace=True)
        x = F.relu(self.bn53(self.conv53(x)), inplace=True)
        x, ind5 = self.pool53(x)

        return x, ind1, ind2, ind3, ind4, ind5


class SegNetDecoder(TurboModule):
    def __init__(self, n_outputs=3, bn_momentum=0.1):
        nn.Module.__init__(self)
        self.n_outputs = n_outputs

        self.conv53d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn53d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv52d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn52d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv51d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn51d = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.conv43d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn43d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv42d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn42d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.conv41d = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn41d = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.conv33d = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn33d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv32d = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn32d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.conv31d = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn31d = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.conv22d = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn22d = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.conv21d = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn21d = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.conv12d = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn12d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv11d = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn11d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv_final = nn.Conv2d(64, n_outputs, 1)

    def forward(self, x, ind1, ind2, ind3, ind4, ind5):
        # Stage 5d
        x = F.max_unpool2d(x, ind5, 2, 2)
        x = F.relu(self.bn53d(self.conv53d(x)), inplace=True)
        x = F.relu(self.bn52d(self.conv52d(x)), inplace=True)
        x = F.relu(self.bn51d(self.conv51d(x)), inplace=True)

        # Stage 4d
        x = F.max_unpool2d(x, ind4, 2, 2)
        x = F.relu(self.bn43d(self.conv43d(x)), inplace=True)
        x = F.relu(self.bn42d(self.conv42d(x)), inplace=True)
        x = F.relu(self.bn41d(self.conv41d(x)), inplace=True)

        # Stage 3d
        x = F.max_unpool2d(x, ind3, 2, 2)
        x = F.relu(self.bn33d(self.conv33d(x)), inplace=True)
        x = F.relu(self.bn32d(self.conv32d(x)), inplace=True)
        x = F.relu(self.bn31d(self.conv31d(x)), inplace=True)

        # Stage 2d
        x = F.max_unpool2d(x, ind2, 2, 2)
        x = F.relu(self.bn22d(self.conv22d(x)), inplace=True)
        x = F.relu(self.bn21d(self.conv21d(x)), inplace=True)

        # Stage 1d
        x = F.max_unpool2d(x, ind1, 2, 2)
        x = F.relu(self.bn12d(self.conv12d(x)), inplace=True)
        x = F.relu(self.bn11d(self.conv11d(x)), inplace=True)

        return self.conv_final(x)


class SegNet(TurboModule):
    def __init__(self, n_inputs=3, n_outputs=3, bn_momentum=0.1):
        nn.Module.__init__(self)
        self.encoder = SegNetEncoder(n_inputs, bn_momentum)
        self.decoder = SegNetDecoder(n_outputs, bn_momentum)

    def finetune(self, state_dict):
        own_state = self.state_dict()
        # copy first 13 layers = 26 params (weights + bias)
        others = state_dict.keys()
        # find best key
        for i, k in enumerate(own_state):
            if 'conv_final' in k:
                continue
            for o in others:
                if k in o:
                    own_state[k] = state_dict[o]
                    break
        self.load_state_dict(own_state)

    def transplant_vgg16(self, vgg_path):
        own_state = self.encoder.state_dict()
        own_state.update(self.decoder.state_dict())
        vgg_state = torch.load(vgg_path)
        # copy first 13 layers = 26 params (weights + bias)
        k = list([k for k in own_state.keys() if 'bn' not in k])[:26]
        for i, (own, vgg) in enumerate(zip(k, vgg_state)):
            own_state[own] = vgg_state[vgg]
        self.encoder.load_state_dict(own_state)
        self.decoder.load_state_dict(own_state)

    def load_state_dict(self, state_dict):
        # enable loading models trained with DataParallel
        state_dict = strip_params(state_dict)
        encoder_dict = filter_state(self.encoder.state_dict(), state_dict)
        self.encoder.load_state_dict(encoder_dict)
        decoder_dict = filter_state(self.decoder.state_dict(), state_dict)
        self.decoder.load_state_dict(decoder_dict)

    def encoder_params(self):
        for _, param in self.encoder.named_parameters():
            yield param

    def decoder_params(self):
        for _, param in self.decoder.named_parameters():
            yield param

    def submodules(self):
        return self.encoder, self.decoder

    def forward(self, x):
        x, ind1, ind2, ind3, ind4, ind5 = self.encoder(x)
        # prevent backward through encoder
        if not self.encoder.training and self.decoder.training:
            x.detach_()
            ind1.detach_()
            ind2.detach_()
            ind3.detach_()
            ind4.detach_()
            ind5.detach_()
        return self.decoder(x, ind1, ind2, ind3, ind4, ind5)


