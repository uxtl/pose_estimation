import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2], # input 112^2 * 16
            [6, 32, 3, 2], # input 56^2 * 24 &
            [6, 64, 4, 2], # input 28^2 * 32 
            [6, 96, 3, 1], # input 28^2 * 64 &
            [6, 160, 3, 2],# input 14^2 * 96 &
            [6, 320, 1, 1],# input 7^2 * 160 
        ]
        # output 7^2 320 &
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features1 = [conv_bn(3, input_channel, 2)]
        self.features2 = []
        self.features3 = []
        self.features4 = []

        # building inverted residual blocks
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[0], self.features1)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[1], self.features1)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[2], self.features2)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[3], self.features2)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[4], self.features3)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[5], self.features4)
        self._make_layer(width_mult, input_channel, self.interverted_residual_setting[6], self.features4)

        # make them nn.Sequential
        self.features1 = nn.Sequential(*self.features1)
        self.features2 = nn.Sequential(*self.features2)
        self.features3 = nn.Sequential(*self.features3)
        self.features4 = nn.Sequential(*self.features4)


        self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        return [x4, x3, x2, x1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def _make_layer(self, width_mult, input_channel, layer_setting, feature):
        for t, c, n, s in layer_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    feature.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    feature.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        return input_channel


def mobile_net_v2(pretrained=False, **kwargs):

    model = MobileNetV2()
    if pretrained:
        # pretrained model / load parameters
        pass
    return model

