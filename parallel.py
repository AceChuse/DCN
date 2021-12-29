#!/usr/bin/python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint_sequential

import math
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

single = _ntuple(1)
pair = _ntuple(2)
triple = _ntuple(3)
quadruple = _ntuple(4)


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class ModuleParall(nn.Module):
    def inner_train(self, mode=True):
        self.inner_training = mode
        for module in self.children():
            if isinstance(module, ModuleParall):
                module.inner_train(mode)
        return self
    def inner_eval(self):
        return self.inner_train(False)
    def repeat_lr(self, lr, wname):
        raise NotImplementedError


class SequentialParall(nn.Sequential):
    def inner_train(self, mode=True):
        self.inner_training = mode
        for module in self.children():
            if isinstance(module, ModuleParall):
                module.inner_train(mode)
            module.inner_train(mode)
        return self

    def inner_eval(self):
        return self.inner_train(False)

    def get_parameters(self, model, num=1):
        raise ValueError('This has not been complete (SequentialParall)!')
        for mP, m in zip(self._modules.values(), model._modules.values()):
            mP.get_parameters(m, num)


def pass_f(m, num=1):
    pass
def pass_t(mode=True):
    pass
def sameparall(m):
    m.get_parameters = pass_f
    m.inner_train = pass_t
    m.inner_eval = pass_t
    return m


class LinearParall(ModuleParall):
    def __init__(self):
        super(LinearParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)

    def forward(self, x):
        output = x.unsqueeze(-2).matmul(self.weight.transpose(2,3)).squeeze(-2)
        if self.bias is not None:
            output += self.bias
        return output

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(self.num, 1, 1).unsqueeze(0)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(self.num, 1).unsqueeze(0)

    def repeat_lr(self, lr, wname):
        if wname == 'weight':
            lr = lr.view(1, self.num, 1, 1)
        elif wname == 'bias':
            lr = lr.view(1, self.num, 1)
        return lr

'''
if __name__ == '__main__':
    x = Variable(torch.arange(0, 24).view(4, 2, 3)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    lnear = nn.Linear(3, 4).cuda()
    lnears = LinearParall()
    lnears.get_parameters(lnear,4)
    k = 1
    print('x=',x)
    print('y1=',lnears(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([lnear(x[i]).unsqueeze(0) for i in range(4)],0)[k])
'''

class Conv2dParall(ModuleParall):
    def __init__(self):
        super(Conv2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if model.groups != 1:
            raise ValueError('model.groups is not equal to 1!')
        self.stride = model.stride
        self.padding = model.padding
        self.dilation = model.dilation
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.num = num

        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)

    def forward(self, x):
        si = x.size()
        x = x.view(si[0], si[1]*si[2], si[3], si[4])
        output = F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])

    def weight_r(self, weight):
        return weight.repeat(self.num, 1, 1, 1)

    def bias_r(self, bias):
        return bias.repeat(self.num)

    def repeat_lr(self, lr, wname):
        lr = lr.view(self.num, 1).repeat(1, self.in_channels)
        if wname == 'weight':
            lr = lr.view(-1, 1, 1, 1)
        elif wname == 'bias':
            lr = lr.view(-1)
        return lr


class Conv2dTParall(ModuleParall):
    def __init__(self):
        super(Conv2dTParall, self).__init__()

    def get_parameters(self, model, num=1):
        if model.groups != 1:
            raise ValueError('model.groups is not equal to 1!')
        self.stride = model.stride
        self.padding = model.padding
        self.output_padding = model.output_padding
        self.dilation = model.dilation
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.num = num

        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)

    def forward(self, x):
        si = x.size()
        x = x.reshape(si[0], si[1]*si[2], si[3], si[4])
        output = F.conv_transpose2d(
            x, self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.num, self.dilation)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])

    def weight_r(self, weight):
        return weight.repeat(self.num, 1, 1, 1)

    def bias_r(self, bias):
        return bias.repeat(self.num)

'''
if __name__ == '__main__':
    #conv = nn.Conv2d(2, 3, 1).cuda()
    conv = nn.ConvTranspose2d(2,3,1).cuda()
    print('weight=', conv.weight)
    print('bias=', conv.bias)
    #convs = Conv2dParall()
    convs = Conv2dTParall()
    convs.get_parameters(conv, 4)
    x = Variable(torch.arange(0, 16 * 9).view(4, 2, 2, 3, 3)).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    k = 0
    print('x=', x)
    print('y1=', convs(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([conv(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class LayerNormParall(ModuleParall):
    def __init__(self):
        super(LayerNormParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.normalized_shape = [num]
        self.re_num = [num]
        self.normalized_shape.extend(list(model.normalized_shape))
        self.len_norm_shape = len(model.normalized_shape)
        self.re_num.extend([1] * self.len_norm_shape)
        self.eps = model.eps
        self.elementwise_affine = model.elementwise_affine

        if self.elementwise_affine:
            if model.weight is None:
                self.weight = None
            else:
                self.weight = model.weight.clone()
                self.weight = self.weight_r(self.weight)

            if model.bias is None:
                self.bias = None
            else:
                self.bias = model.bias.clone()
                self.bias = self.bias_r(self.bias)

    def forward(self, x):
        in_size = x.size()
        re_size = list(in_size[:-self.len_norm_shape])
        re_size.append(-1)
        x = x.view(*re_size)
        x = (x - x.mean(-1, keepdim=True).detach()) / \
                torch.sqrt(x.var(-1, keepdim=True, unbiased=False).detach() + self.eps)
        x = x.view(in_size)
        normalized_shape = [1] + self.normalized_shape[:1] + [1] * (
            len(in_size) - self.len_norm_shape - 2) + self.normalized_shape[1:]

        if self.elementwise_affine:
            weight = self.weight.view(normalized_shape)
            bias = self.bias.view(normalized_shape)
            return x * weight + bias
        else:
            return x

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(*self.re_num)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(*self.re_num)

'''
if __name__ == '__main__':
    x = torch.randn(4, 3, 2, 2).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    ln = nn.LayerNorm([2,2]).cuda()
    lns = LayerNormParall()
    lns.get_parameters(ln,num=4)
    k = 3
    print('x=', x)
    print('y1=', lns(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ln(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class BatchNormParall(ModuleParall):
    def __init__(self):
        super(BatchNormParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.num = num
        self.num_features = model.num_features
        self.eps = model.eps
        self.momentum = model.momentum
        self.affine = model.affine
        self.training = model.training
        self.track_running_stats = model.track_running_stats

        if self.affine:
            self.weight = model.weight.clone()
            self.weight = self.weight_r(self.weight)
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            raise ValueError('track_running_stats is True!')
            self.running_mean = model.running_mean
            self.running_mean = self.running_mean.repeat(num)
            self.running_var = model.running_var
            self.running_var = self.running_var.repeat(num)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        in_size = list(x.size())
        re_size = in_size[0:1] + [in_size[1] * in_size[2]] + in_size[3:]
        x = x.view(*re_size)
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps).view(in_size)

    def weight_r(self, weight):
        return weight.repeat(self.num)

    def bias_r(self, bias):
        return bias.repeat(self.num)

    def repeat_lr(self, lr, wname):
        lr = lr.view(self.num, 1).repeat(1, self.num_features)
        lr = lr.view(-1)
        return lr


class InstanceNormParall(BatchNormParall):
    def __init__(self):
        super(InstanceNormParall, self).__init__()

    def forward(self, x):
        in_size = list(x.size())
        re_size = in_size[0:1] + [in_size[1] * in_size[2]] + in_size[3:]
        x = x.view(*re_size)
        return F.instance_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps).view(in_size)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 2, 2).type(FloatTensor)
    #x = torch.randn(4, 3, 2, 2).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    #bn = nn.BatchNorm2d(2).cuda()
    #bn = nn.BatchNorm1d(2).cuda()
    bn = nn.InstanceNorm2d(2, affine=True).cuda()
    #bn = nn.InstanceNorm1d(2, affine=True).cuda()
    #bn.eval()
    #bns = BatchNormParall()
    bns = InstanceNormParall()
    bns.get_parameters(bn,num=4)
    k = 2
    print('x=', x)
    print('y1=', bns(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([bn(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class AvgPool2dParall(ModuleParall):
    def __init__(self):
        super(AvgPool2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if isinstance(model.kernel_size, int):
            self.kernel_size = (1, model.kernel_size, model.kernel_size)
        else:
            self.kernel_size = (1, model.kernel_size[0], model.kernel_size[1])
        if isinstance(model.stride, int):
            self.stride = (1, model.stride, model.stride)
        else:
            self.stride = (1, model.stride[0], model.stride[1])
        self.padding = model.padding
        self.ceil_mode = model.ceil_mode
        self.count_include_pad = model.count_include_pad

    def forward(self, x):
        return F.avg_pool3d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)


class MaxPool2dParall(ModuleParall):
    def __init__(self):
        super(MaxPool2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if isinstance(model.kernel_size, int):
            self.kernel_size = (1, model.kernel_size, model.kernel_size)
        else:
            self.kernel_size = (1, model.kernel_size[0], model.kernel_size[1])
        if isinstance(model.stride, int):
            self.stride = (1, model.stride, model.stride)
        else:
            self.stride = (1, model.stride[0], model.stride[1])
        self.padding = model.padding
        self.dilation = model.dilation
        self.return_indices = model.return_indices
        self.ceil_mode = model.ceil_mode

    def forward(self, x):
        return F.max_pool3d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 4, 4).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    #ap = nn.AvgPool2d(2).cuda()
    ap = nn.MaxPool2d(2).cuda()
    #aps = AvgPool2dParall()
    aps = MaxPool2dParall()
    aps.get_parameters(ap,num=4)
    k = 2
    print('x=', x)
    print('y1=', aps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ap(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class AvgPool1dParall(ModuleParall):
    def __init__(self):
        super(AvgPool1dParall, self).__init__()

    def get_parameters(self, model, num=1):
        if len(model.kernel_size) == 1:
            self.kernel_size = (1, model.kernel_size[0])
        else:
            raise ValueError('kernel_size should be 1 dim.')
        if len(model.stride) == 1:
            self.stride = (1, model.stride[0])
        else:
            raise ValueError('stride should be 1 dim.')
        self.padding = model.padding
        self.ceil_mode = model.ceil_mode
        self.count_include_pad = model.count_include_pad

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 3, 2, 4).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    ap = nn.AvgPool1d(2).cuda()
    #ap = nn.MaxPool2d(2).cuda()
    aps = AvgPool1dParall()
    #aps = MaxPool2dParall()
    aps.get_parameters(ap,num=4)
    k = 2
    print('x=', x)
    print('y1=', aps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([ap(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

def squash(x, dim=1, eps=1e-5):
    x_norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    x_norm2 = x_norm ** 2
    return x_norm2 / (1 + x_norm2) * x / (x_norm + eps)


class Capsule(ModuleParall):
    def __init__(self, in_feature, out_feature, routings=3, bias=True, retain_grad=False):
        super(Capsule, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.routings = routings
        if routings < 1:
            raise ValueError('Routing should be at least 1!')
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, self.out_feature)).type(FloatTensor)
        self.c = F.softmax(b, 2)
        self.weight = Parameter(torch.Tensor(self.in_feature, self.out_feature))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_feature))
        else:
            self.bias = None
            self.forward = self.no_bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u, bias = (u_hat, self.bias) if self.retain_grad else (u_hat.detach(), self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1) + bias, dim=1)
            b = b + u * v.view(-1, 1, self.out_feature)
            c = F.softmax(b, 2)
        return squash(torch.sum(u_hat * c, dim=1) + self.bias, dim=1)

    def no_bias(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u = u_hat if self.retain_grad else u_hat.detach()
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1), dim=1)
            b = b + u * v.view(-1, 1, self.out_feature)
            c = F.softmax(b, 2)
        return squash(torch.sum(u_hat * c, dim=1), dim=1)

    def extra_repr(self):
        s = ('{in_feature}, {out_feature}, routings={routings}'
             ', bias='+str(self.bias is not None)+', retain_grad={retain_grad}')
        return s.format(**self.__dict__)


class CapsuleShare(ModuleParall):
    def __init__(self, in_feature, out_feature, routings=3, bias=True, retain_grad=False):
        super(CapsuleShare, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.routings = routings
        if routings < 1:
            raise ValueError('Routing should be at least 1!')
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, self.out_feature)).type(FloatTensor)
        self.c = F.softmax(b, 2)
        self.weight = Parameter(torch.Tensor(self.in_feature, 1, self.out_feature))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_feature))
        else:
            self.bias = None
            self.forward = self.no_bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u_hat = u_hat.view(-1,self.in_feature * u.size(-1),self.out_feature)
        u, bias = (u_hat,self.bias) if self.retain_grad else (u_hat.detach(),self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1) + bias, dim=1)
            b = b + u * v.view(-1, 1, self.out_feature)
            c = F.softmax(b, 2)
        v = squash(torch.sum(u_hat * c, dim=1) + self.bias, dim=1)
        return v

    def no_bias(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u_hat = u_hat.view(-1,self.in_feature * u.size(-1),self.out_feature)
        u = u_hat if self.retain_grad else u_hat.detach()
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1), dim=1)
            b = b + u * v.view(-1, 1, self.out_feature)
            c = F.softmax(b, 2)
        v = squash(torch.sum(u_hat * c, dim=1), dim=1)
        return v

    def extra_repr(self):
        s = ('{in_feature}, {out_feature}, routings={routings}'
             ', bias='+str(self.bias is not None)+', retain_grad={retain_grad}')
        return s.format(**self.__dict__)

'''
class CapsuleConv2d(ModuleParall):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, routings=3, retain_grad=False):
        super(CapsuleConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.routings = routings
        if routings < 1:
            raise ValueError('Routing should be at least 1!')
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, self.out_channels)).type(FloatTensor)
        self.c = F.softmax(b, 2)

        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)

        self.kernel_size = kernel_size
        self.k = 1
        for k in kernel_size:
            self.k *= k
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
            in_channels, self.k, 1, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, u):
        batch_size = u.size(0)
        c = self.c
        u = F.unfold(u, self.kernel_size, self.dilation, self.padding, self.stride
                     ).view(batch_size, self.in_channels, self.k, -1, 1)
        u_hat = torch.sum(u * self.weight, dim=2).view(batch_size, -1, self.out_channels)
        u, bias = (u_hat, self.bias) if self.retain_grad else (u_hat.detach(), self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1) + bias, dim=1)
            b = b + u * v.view(-1, 1, self.out_channels)
            c = F.softmax(b, 2)
        return squash(torch.sum(u_hat * c + bias, dim=1), dim=1)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, routings={routings}'
             ', retain_grad={retain_grad}')
        return s.format(**self.__dict__)
'''

class CapsuleParall(ModuleParall):
    def __init__(self):
        super(CapsuleParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.in_feature = model.in_feature
        self.out_feature = model.out_feature
        self.routings = model.routings
        self.retain_grad = model.retain_grad
        self.num = num

        self.c = model.c.unsqueeze(0)
        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is not None:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)
        else:
            self.forward = self.no_bias

    def forward(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u, bias = (u_hat, self.bias) if self.retain_grad else (u_hat.detach(),self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=2) + bias, dim=2)
            b = b + u * v.view(-1, self.num, 1, self.out_feature)
            c = F.softmax(b, 3)
        return squash(torch.sum(u_hat * c, dim=2) + self.bias, dim=2)

    def no_bias(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u = u_hat if self.retain_grad else u_hat.detach()
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=2), dim=2)
            b = b + u * v.view(-1, self.num, 1, self.out_feature)
            c = F.softmax(b, 3)
        return squash(torch.sum(u_hat * c, dim=2), dim=2)

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(self.num,1,1).unsqueeze(0)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(self.num,1).unsqueeze(0)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = Variable(torch.arange(0, 24).view(4, 2, 3)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    cs = Capsule(3,4,bias=False).cuda()
    print('weight=',cs.weight)
    css = CapsuleParall()
    css.get_parameters(cs,4)
    k = 1
    print('x=',x)
    print('y1=',css(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([cs(x[i]).unsqueeze(0) for i in range(4)],0)[k])
'''

class CapsuleSParall(ModuleParall):
    def __init__(self):
        super(CapsuleSParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.in_feature = model.in_feature
        self.out_feature = model.out_feature
        self.routings = model.routings
        self.retain_grad = model.retain_grad
        self.num = num

        self.c = model.c.unsqueeze(0)
        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is not None:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)
        else:
            self.forward = self.no_bias

    def forward(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u_hat = u_hat.view(-1, self.num, self.in_feature * u.size(-1), self.out_feature)
        u, bias = (u_hat, self.bias) if self.retain_grad else (u_hat.detach(), self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=2) + bias, dim=2)
            b = b + u * v.view(-1, self.num, 1, self.out_feature)
            c = F.softmax(b, 3)
        v = squash(torch.sum(u_hat * c, dim=2) + self.bias, dim=2)
        return v

    def no_bias(self, u):
        c = self.c
        u_hat = u.unsqueeze(-1) * self.weight
        u_hat = u_hat.view(-1, self.num, self.in_feature * u.size(-1), self.out_feature)
        u = u_hat if self.retain_grad else u_hat.detach()
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=2), dim=2)
            b = b + u * v.view(-1, self.num, 1, self.out_feature)
            c = F.softmax(b, 3)
        v = squash(torch.sum(u_hat * c, dim=2), dim=2)
        return v

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(self.num,1,1,1).unsqueeze(0)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(self.num,1).unsqueeze(0)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = Variable(torch.arange(0, 48).view(4, 2, 3, 2)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    cs = CapsuleShare(3, 4, bias=False).cuda()
    print('weight=',cs.weight)
    css = CapsuleSParall()
    num = 3
    css.get_parameters(cs, num)
    k = 1
    print('x=',x)
    print('y1=',css(xt[:,:num]).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([cs(x[i]).unsqueeze(0) for i in range(4)],0)[k])
'''
'''
class CapsuleC2dParall(Conv2dParall):
    def __init__(self):
        super(CapsuleC2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)
        self.in_channels = model.in_channels
        self.kernel_size = model.kernel_size
        self.k = model.k
        self.routings = model.routings
        self.retain_grad = model.retain_grad
        self.c = model.c.unsqueeze(0)

    def forward(self, u):
        si = u.size()
        c = self.c
        u = u.view(si[0] * si[1], si[2], si[3], si[4])
        u = F.unfold(u, self.kernel_size, self.dilation, self.padding, self.stride
                     ).view(si[0], self.num, self.in_channels, self.k, -1, 1)
        u_hat = torch.sum(u * self.weight, dim=3).view(si[0], self.num, -1, self.out_channels)
        u, bias = (u_hat, self.bias) if self.retain_grad else (u_hat.detach(), self.bias.detach())
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=2) + bias, dim=2)
            b = b + u * v.view(-1, self.num, 1, self.out_channels)
            c = F.softmax(b, 3)
        v = squash(torch.sum(u_hat * c, dim=2) + self.bias, dim=2)
        return v

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(self.num, 1, 1, 1, 1)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(self.num, 1)
'''
'''
if __name__ == '__main__':
    conv = CapsuleConv2d(2, 3, 1).cuda()
    print('weight=', conv.weight.size())
    print('bias=', conv.bias.size())
    convs = CapsuleC2dParall()
    convs.get_parameters(conv, 4)
    x = Variable(torch.arange(0, 16 * 9).view(4, 2, 2, 3, 3)).type(FloatTensor)
    xt = x.transpose(0,1).contiguous()
    k = 0
    print('y1=', convs(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([conv(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
'''

class CapsuleCo(ModuleParall):
    # TODO: The correct vesion of Capsule!
    def __init__(self, in_vecn, in_vecl, out_vecn, out_vecl, routings=3, bias=True, share=False, retain_grad=False):
        super(CapsuleCo, self).__init__()
        self.in_vecn = in_vecn
        self.in_vecl = in_vecl
        self.in_feature = in_vecn * in_vecl
        self.out_vecn = out_vecn
        self.out_vecl = out_vecl
        self.out_feature = out_vecn * out_vecl
        self.routings = routings
        if routings < 1:
            raise ValueError('Routing should be at least 1!')
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, out_vecn, 1)).type(FloatTensor)
        self.c = F.softmax(b, 2)
        n = 1 if share else in_vecn
        self.weight = Parameter(torch.Tensor(n, out_vecn, out_vecl, in_vecl))
        self.bias = Parameter(torch.Tensor(n, out_vecn, out_vecl)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_feature)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, u):
        # c.size()=[1, 1, out_vecn, 1]
        c = self.c
        # u.size()=[batch_size, in_vecn, in_vecl]
        # self.weight.size()=[in_vecn, out_vecn, out_vecl, in_vecl]
        # self.bias.size()=[in_vecn, out_vecn, out_vecl]
        u_hat = u.unsqueeze(-2).unsqueeze(-2).matmul(self.weight.transpose(-2, -1)).squeeze(-2)
        u_hat = u_hat + self.bias if self.bias is not None else u_hat
        u = u_hat if self.retain_grad else u_hat.detach()
        # u_hat.size()=[batch_size, in_vecn, out_vecn, out_vecl]
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=-3), dim=-1)
            # v.size()=[batch_size, out_vecn, out_vecl]
            b = b + torch.sum(u * v.unsqueeze(-3), dim=-1, keepdim=True)
            # b.size()=[batch_size, in_vecn, out_vecn, 1]
            c = F.softmax(b, -2)
        return squash(torch.sum(u_hat * c, dim=-3), dim=-1)

    def extra_repr(self):
        s = ('in=({in_vecn}, {in_vecl}), out=({out_vecn}, {out_vecl}), routings={routings}'
             ', bias='+str(self.bias is not None)+', retain_grad={retain_grad}')
        return s.format(**self.__dict__)


class CapsuleCoParall(ModuleParall):
    # TODO: The correct vesion of Capsule!
    def __init__(self):
        super(CapsuleCoParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.in_vecn = model.in_vecn
        self.in_vecl = model.in_vecl
        self.in_feature = model.in_feature
        self.out_vecn = model.out_vecn
        self.out_vecl = model.out_vecl
        self.out_feature = model.out_feature
        self.routings = model.routings
        self.retain_grad = model.retain_grad
        self.num = num

        self.c = model.c.unsqueeze(0)
        self.weight = model.weight.clone()
        self.weight = self.weight_r(self.weight)
        if model.bias is not None:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)
        else:
            self.bias = model.bias

    def forward(self, u):
        # c.size()=[1, 1, 1, out_vecn, 1]
        c = self.c
        # u.size()=[batch_size, meta_batch_size, in_vecn, in_vecl]
        # self.weight.size()=[1, meta_batch_size, in_vecn, out_vecn, out_vecl, in_vecl]
        # self.bias.size()=[1, meta_batch_size, in_vecn, out_vecn, out_vecl]
        u_hat = u.unsqueeze(-2).unsqueeze(-2).matmul(self.weight.transpose(-2, -1)).squeeze(-2)
        u_hat = u_hat + self.bias if self.bias is not None else u_hat
        # u_hat.size()=[batch_size, meta_batch_size, in_vecn, out_vecn, out_vecl]
        u = u_hat if self.retain_grad else u_hat.detach()
        b = 0.
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=-3), dim=-1)
            # v.size()=[batch_size, meta_batch_size, out_vecn, out_vecl]
            b = b + torch.sum(u * v.unsqueeze(-3), dim=-1, keepdim=True)
            # b.size()=[batch_size, meta_batch_size, in_vecn, out_vecn, 1]
            c = F.softmax(b, -2)
        return squash(torch.sum(u_hat * c, dim=-3), dim=-1)

    def weight_r(self, weight):
        return weight.unsqueeze(0).repeat(self.num,1,1,1,1).unsqueeze(0)

    def bias_r(self, bias):
        return bias.unsqueeze(0).repeat(self.num,1,1,1).unsqueeze(0)

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = Variable(torch.arange(0, 4 * 2 * 5 * 3).view(4, 2, 5, 3)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    cs = CapsuleCo(in_vecn=5, in_vecl=3, out_vecn=4, out_vecl=2,bias=True,share=True).cuda()
    print('weight=',cs.weight)
    css = CapsuleCoParall()
    css.get_parameters(cs,4)
    k = 2
    print('y2=', torch.cat([cs(x[i]).unsqueeze(0) for i in range(4)], 0)[k])
    print('y1=',css(xt).transpose(0,1).contiguous()[k])
'''

class Norm(ModuleParall):
    def __init__(self, p=2, dim=-1, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        s = ('p={p}, dim={dim}, keepdim={keepdim}')
        return s.format(**self.__dict__)


class DropoutParall(ModuleParall):
    def __init__(self):
        super(DropoutParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.call = (model,)

    def forward(self, x):
        in_size = list(x.size())
        re_size = [in_size[0]*in_size[1]] + in_size[2:]
        x = x.view(*re_size)
        return self.call[0](x).view(*in_size)

    def inner_train(self, mode=True):
        if self.training:
            self.call[0].train(not mode)
        else:
            self.call[0].eval()
        self.inner_training = mode
        for module in self.children():
            module.inner_train(mode)
        return self

'''
if __name__ == '__main__':
    torch.manual_seed(0)
    x = Variable(torch.arange(0, 48*3).view(4, 2, 3, 2, 3)).type(FloatTensor)
    xt = x.transpose(0, 1).contiguous()
    dp = nn.Dropout2d(p=0.2)
    dps = DropoutParall()
    dps.get_parameters(dp,4)
    k = 1
    print('x=',x)
    print('y1=',dps(xt).transpose(0,1).contiguous()[k])
    print('y2=', torch.cat([dp(x[i]).unsqueeze(0) for i in range(4)],0))
'''

class MetaDropout(ModuleParall):
    def __init__(self, in_channels, p=0.5, p_last=None, t_max=None, mode=None):
        super(MetaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.in_channels = in_channels
        self.p = p
        self.q = 1. - p
        self.q_cur = self.q
        self.p_last = p_last
        if self.p_last is not None:
            self.q_last = 1. - self.p_last
        self.t_max = t_max
        self.t_cur = 0
        self.mode = mode
        if mode == 'CosineAnnealing':
            self.new_q = self.cosine_annealing
        else:
            self.new_q = self.pass_q

    def pass_q(self):
        pass

    def cosine_annealing(self):
        self.q_cur = self.q_last + (self.q - self.q_last
                                    ) / 2. * (1. + math.cos(self.t_cur / self.t_max))
        self.t_cur = self.t_cur + 1 if self.t_cur < self.t_max else self.t_cur

    def extra_repr(self):
        return 'p={}, p_last={}, t_max={}, mode={}'.format(
            self.p, self.p_last, self.t_max, self.mode)

    def reset(self, num):
        if self.training:
            self.new_q()
        self.out = torch.full((num, self.in_channels,), self.q_cur)
        self.out = self.out.bernoulli() / self.q_cur
        self.out = self.out.type(FloatTensor)

    def forward(self, x):
        if self.training:
            in_size = list(x.size())
            re_out = [1,in_size[1],self.in_channels] + (len(in_size) - 3) * [1]
            return x * self.out.view(*re_out)
        else:
            return x

'''
if __name__ == '__main__':
    torch.manual_seed(3)
    x = Variable(torch.arange(0, 144).view(3, 2, 4, 2, 3)).type(FloatTensor)
    xt = x
    dp = MetaDropout(4, p=0.5)
    dp.reset(2)

    k = 2
    print('y1=',dp(xt))
'''

class Block2d(nn.Module):
   expansion = 1

   def __init__(self, inplanes, planes, stride=1, downsample=None):
       super(Block2d, self).__init__()
       self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                    padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)
       self.downsample = downsample
       self.stride = stride

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)

       if self.downsample is not None:
           residual = self.downsample(x)

       out += residual
       out = self.relu(out)

       return out


class Block2dDP(Block2d):

   def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
       super(Block2dDP, self).__init__(inplanes, planes, stride, downsample)
       self.dp = nn.Dropout2d(p=kwargs['p'])

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)
       out = self.dp(out)

       out = self.conv2(out)
       out = self.bn2(out)

       if self.downsample is not None:
           residual = self.downsample(x)

       out += residual
       out = self.relu(out)

       return out


class Block2d313(nn.Module):
   expansion = 1

   def __init__(self, inplanes, planes, stride=1, downsample=None):
       super(Block2d313, self).__init__()
       self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                              padding=0, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)
       self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
                    padding=1, bias=False)
       self.bn3 = nn.BatchNorm2d(planes)
       self.downsample = downsample
       self.stride = stride

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)
       out = self.relu(out)

       out = self.conv3(out)
       out = self.bn3(out)

       if self.downsample is not None:
           residual = self.downsample(x)

       out += residual
       out = self.relu(out)

       return out


class Block2dN313(Block2d313):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block2dN313, self).__init__(inplanes, planes, stride, downsample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class MakeResnet2d(nn.Module):

    def __init__(self, block, inplanes, planes, blocks, stride=1, **kwargs):
        super(MakeResnet2d, self).__init__()
        self.block = block
        self.input_size = inplanes
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.add_module(str(0), block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.add_module(str(i), block(self.inplanes, planes, **kwargs))

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + self.block.__name__ + ',' \
            + str(int(self.input_size)) + ',' \
            + str(self.planes) + ',' \
            + 'blocks=' + str(self.blocks) + ',' \
            + 'stride=' + str(self.stride) + ')'


class BlockT2d(nn.Module):
   expansion = 1

   def __init__(self, inplanes, planes, stride=1, output_padding=1, downsample=None):
       super(BlockT2d, self).__init__()
       self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=stride,
                                       padding=1, output_padding=output_padding, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3,
                    padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)
       self.downsample = downsample
       self.stride = stride

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)

       if self.downsample is not None:
           residual = self.downsample(x)

       out += residual
       out = self.relu(out)

       return out


class BlockT2d313(nn.Module):
   expansion = 1

   def __init__(self, inplanes, planes, stride=1, output_padding=1, downsample=None):
       super(BlockT2d313, self).__init__()
       self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=stride,
                                       padding=1, output_padding=output_padding, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=1,
                                       padding=0, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)
       self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=3,
                    padding=1, bias=False)
       self.bn3 = nn.BatchNorm2d(planes)
       self.downsample = downsample
       self.stride = stride

   def forward(self, x):
       residual = x

       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)

       out = self.conv2(out)
       out = self.bn2(out)
       out = self.relu(out)

       out = self.conv3(out)
       out = self.bn3(out)

       if self.downsample is not None:
           residual = self.downsample(x)

       out += residual
       out = self.relu(out)

       return out


class MakeResnetT2d(nn.Module):

    def __init__(self, block, inplanes, planes, blocks, stride=1, **kwargs):
        super(MakeResnetT2d, self).__init__()
        self.block = block
        self.input_size = inplanes
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        self.output_padding = kwargs.get('output_padding', 1)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                       kernel_size=3, stride=stride, padding=1,
                                       output_padding=self.output_padding, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                       kernel_size=3, stride=stride, padding=1,
                                       output_padding=self.output_padding, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        self.add_module(str(0), block(
            self.inplanes, planes, stride, downsample=downsample,
            output_padding=self.output_padding))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.add_module(str(i), block(self.inplanes, planes, output_padding=0))

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + self.block.__name__ + ',' \
            + str(int(self.input_size)) + ',' \
            + str(self.planes) + ',' \
            + 'blocks=' + str(self.blocks) + ',' \
            + 'stride=' + str(self.stride) + ')'


class NonFineTune(nn.Sequential):
    def forward(self, x):
        in_size = list(x.size())
        re_size = [in_size[0] * in_size[1]] + in_size[2:]
        x = x.view(*re_size)
        for module in self._modules.values():
            x = module(x)
        out_size = list(x.size())
        re_size = [in_size[0], in_size[1]] + out_size[1:]
        return x.view(*re_size)


class NonFineTuneCP(NonFineTune):
    def forward(self, x):
        in_size = list(x.size())
        re_size = [in_size[0] * in_size[1]] + in_size[2:]
        x = x.view(*re_size)
        x.requires_grad = True
        y = checkpoint_sequential(self, 2, x)
        x.requires_grad = False
        out_size = list(y.size())
        re_size = [in_size[0], in_size[1]] + out_size[1:]
        return y.view(*re_size)


class NFTModule(nn.Module):
    def __init__(self):
        super(NFTModule, self).__init__()


class NFTSequential(NFTModule, nn.Sequential):
    def __init__(self, *args):
        nn.Sequential.__init__(self, *args)