import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn
from numpy import multiply

def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        elif name == 'insnorm':
            nonlinear.add_module('insnorm', nn.InstanceNorm1d(channels))
        elif name == 'GW':
            nonlinear.add_module('GW', GroupItN(channels,1,num_channels=0, T=5,dim=3, affine=False, momentum=1.))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def high_order_statistics_pooling(x,
                                  dim=-1,
                                  keepdim=False,
                                  unbiased=True,
                                  eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    norm = (x - mean.unsqueeze(dim=dim)) \
        / std.clamp(min=eps).unsqueeze(dim=dim)
    skewness = norm.pow(3).mean(dim=dim)
    kurtosis = norm.pow(4).mean(dim=dim)
    stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats




class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class HighOrderStatsPool(nn.Module):
    def forward(self, x):
        return high_order_statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        # print('outsize:',x.shape)
        return x


class DenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = nn.Conv1d(bn_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.linear2(self.nonlinear2(x))
        return x


class DenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                   out_channels=out_channels,
                                   bn_channels=bn_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class StatsSelect(nn.Module):
    def __init__(self, channels, branches, null=False, reduction=1):
        super(StatsSelect, self).__init__()
        self.gather = HighOrderStatsPool()
        self.linear1 = nn.Conv1d(channels * 4, channels // reduction, 1)
        self.linear2 = nn.ModuleList()
        if null:
            branches += 1
        for _ in range(branches):
            self.linear2.append(nn.Conv1d(channels // reduction, channels, 1))
        self.channels = channels
        self.branches = branches
        self.null = null
        self.reduction = reduction

    def forward(self, x):
        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        x = torch.sum(f, dim=1)
        x = self.linear1(self.gather(x).unsqueeze(dim=-1))
        s = []
        for linear in self.linear2:
            s.append(linear(x).view(-1, 1, self.channels))
        s = torch.cat(s, dim=1)
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        if self.null:
            s = s[:, :-1, :, :]
        return torch.sum(f * s, dim=1)

    def extra_repr(self):
        return 'channels={}, branches={}, reduction={}'.format(
            self.channels, self.branches, self.reduction)


class MultiBranchDenseTDNNLayer(DenseTDNNLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=(1, ),
                 bias=False,
                 null=False,
                 reduction=1,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2
        if not isinstance(dilation, (tuple, list)):
            dilation = (dilation, )
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = nn.ModuleList()
        for _dilation in dilation:
            self.linear2.append(
                nn.Conv1d(bn_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding * _dilation,
                          dilation=_dilation,
                          bias=bias))
        self.select = StatsSelect(out_channels,
                                  len(dilation),
                                  null=null,
                                  reduction=reduction)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.nonlinear2(x)
        x = self.select([linear(x) for linear in self.linear2])
        return x


class MultiBranchDenseTDNNBlock(DenseTDNNBlock):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 null=False,
                 reduction=1,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = MultiBranchDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                null=null,
                reduction=reduction,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.l1 = nn.Linear(251, 512)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=512,  # RNN隐藏神经元个数
            num_layers=3,  # RNN隐藏层个数
            batch_first = False
        )
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, 251)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.l1(x)
        # print('x:',x.shape)
        # x = x.transpose(1,2)
        out, (h, c) = self.rnn(x)
        # out = out.transpose(1,2)
        # print('out:',out.shape)
        out = self.relu(self.bn(self.out(out)))
        # print('out:',out.shape)
        # out = torch.mean(out, dim=1)
        # print('out:',out.shape)
        # print('h:',h)
        # print('c:',c)
        return out
        # return out, (h, c)

class LSTM2(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.l1 = nn.Linear(251, 128)
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,  # RNN隐藏神经元个数
            num_layers=3,  # RNN隐藏层个数
            batch_first = False
        )
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 128)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.l1(x)
        # print('x:',x.shape)
        # x = x.transpose(1,2)
        out, (h, c) = self.rnn(x)
        # out = out.transpose(1,2)
        # print('out:',out.shape)
        out = self.relu(self.bn(self.out(out)))
        # print('out:',out.shape)
        # out = torch.mean(out, dim=1)
        # print('out:',out.shape)
        # print('h:',h)
        # print('c:',c)
        return out