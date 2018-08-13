import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet
from .MobileNetV2 import *
from .resnet import *

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, extract_net, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [320, 96, 64, 24]#[2048, 1024, 512, 256]
        self.extract_net = extract_net
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.extract_net(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model
    

def CPN_mobilev2(out_size,num_class,pretrained=True):
    mobile_v2 = mobile_net_v2(pretrained=pretrained)
    model = CPN(mobile_v2, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model
