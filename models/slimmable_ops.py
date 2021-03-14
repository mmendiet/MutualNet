import torch.nn as nn
from utils.config import FLAGS

def gen_subnet_cfgs():
    c = []
    prune_chnls = FLAGS.subnet_chnls
    for subnetidx in range(len(prune_chnls)):
        subnet_chnls = prune_chnls[subnetidx]
        subcfg = [[3, subnet_chnls[0]]]
        for i in range(len(subnet_chnls) - 1):
            io_channels = [subnet_chnls[i], subnet_chnls[i + 1]]
            subcfg.append(io_channels)
        subcfg.append([subnet_chnls[-1], subnet_chnls[-1]])
        c.append(subcfg)
    return c

cfgs = gen_subnet_cfgs()

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True):
        groups = in_channels if depthwise else 1
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.subnet_idx = 0
        self.layer_idx = 0
        self.shortcut = False

    def forward(self, input):
        cfg = cfgs[self.subnet_idx]
        if self.shortcut:
            if FLAGS.depth==50:
                shift = 2
            else:
                shift = 1
            in_channels = cfg[self.layer_idx-shift][0]
            out_channels = cfg[self.layer_idx][1]
        else:
            in_channels = cfg[self.layer_idx][0]
            out_channels = cfg[self.layer_idx][1]
        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.subnet_idx = 0
        self.layer_idx = 0

    def forward(self, input):
        cfg = cfgs[self.subnet_idx]
        in_features = cfg[-1][1]
        weight = self.weight[:self.out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(USBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.resobn = nn.ModuleList(
            [nn.BatchNorm2d(self.num_features) for i in range(len(FLAGS.width_list))])
        self.subnet_idx = 0
        self.reso_idx = 0
        self.layer_idx = 0
        self.ignore_model_profiling = True

    def forward(self, input):
        cfg = cfgs[self.subnet_idx]
        c = cfg[self.layer_idx][1]
        weight = self.resobn[self.reso_idx].weight
        bias = self.resobn[self.reso_idx].bias
        y = nn.functional.batch_norm(
            input,
            self.resobn[self.reso_idx].running_mean[:c],
            self.resobn[self.reso_idx].running_var[:c],
            weight[:c],
            bias[:c],
            self.resobn[self.reso_idx].training,
            self.resobn[self.reso_idx].momentum,
            self.resobn[self.reso_idx].eps)
        return y
