from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def ca_activation_func(x, t=1):
    return - torch.threshold(-torch.relu(x), threshold=-t, value=-t)


class CALayerAbs(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CALayerAbs, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, weight=self.weight.abs(), bias=(- self.bias.abs()))
        else:
            return nn.functional.linear(input, weight=self.weight.abs(), bias=None)

    def transform_weights(self):
        self.weight.data.abs_()
        if self.bias is not None:
            self.bias.data = - self.bias.data.abs()


class CALayerAbsProjected(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CALayerAbsProjected, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        self.transform_weights()
        if self.bias is not None:
            return nn.functional.linear(input, weight=self.weight, bias=self.bias)
        else:
            return nn.functional.linear(input, weight=self.weight, bias=None)

    def transform_weights(self):
        self.weight.data.abs_()
        if self.bias is not None:
            self.bias.data = - self.bias.data.abs()


class CALayerReLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CALayerReLU, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(
                input, weight=self.weight.clamp(min=0), bias=self.bias.clamp(max=0))
        else:
            return nn.functional.linear(input, weight=torch.relu(self.weight), bias=None)

    def transform_weights(self):
        self.weight.data.clamp_(min=0)
        if self.bias is not None:
            self.bias.data.clamp_(max=0)


class CALayerReLUProjected(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CALayerReLUProjected, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        self.transform_weights()
        if self.bias is not None:
            return nn.functional.linear(input, weight=self.weight, bias=self.bias)
        else:
            return nn.functional.linear(input, weight=self.weight, bias=None)

    def transform_weights(self):
        self.weight.data.clamp_(min=0)
        if self.bias is not None:
            self.bias.data.clamp_(max=0)
