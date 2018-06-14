from __future__ import print_function, division

from collections import OrderedDict
from abc import ABCMeta

from six import text_type
from six import binary_type

from torch import nn
from torch.nn import init


CONTAINER_NAMES = (
    'module.',
    'segnet.',
    'resnet.',
    'encoder.',
    'decoder.',
    'classifier.',
)


class TurboModule(nn.Module):
    __metaclass__ = ABCMeta

    def init_xavier(self):
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform(m.weight)
                init.constant(m.bias, 0)


def strip_modules(s, *modules):
    stripped = []
    for m in modules:
        _, m, s = s.rpartition(m)
        stripped.append(m)
    return s, stripped


def get_params(state):
    return state.get('model_state', state)


def get_tensors(params):
    return OrderedDict(
        (k, (v.data if isinstance(v, nn.Parameter) else v).cpu())
        for k, v in params.items()
    )


def strip_params(params, modules=CONTAINER_NAMES):
    return OrderedDict(
        (strip_modules(k, *modules)[0], p)
        for k, p in params.items()
    )


def other_type(s):
    if isinstance(s, text_type):
        return s.encode('utf-8')
    elif isinstance(s, binary_type):
        return s.decode('utf-8')


def try_dicts(k, *ds):
    for d in ds:
        v = d.get(k)
        if v is not None:
            return v
    raise KeyError(k)


def try_types(k, *ds):
    try:
        return try_dicts(k, *ds)
    except KeyError:
        return try_dicts(other_type(k), *ds)


def filter_state(own_state, state_dict):
    return OrderedDict((k, try_types(k, state_dict, own_state))
                       for k in own_state)
