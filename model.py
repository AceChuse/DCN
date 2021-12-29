#!/usr/bin/python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

import numpy as np
import math
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class _ConvNdA(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdA, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        #self.rls = []
        self.thetas_init = []
        self.poses_w = []
        self.prod_ks = np.prod(kernel_size)
        if self.prod_ks > 1:
            self.w_k = Parameter(torch.Tensor(1, out_channels, 1, 1, self.prod_ks))
            self.b_k = Parameter(torch.Tensor(1, out_channels, 1, 1, self.prod_ks))
            #self.rl_k = Parameter(torch.Tensor(num_grad, out_channels, 1, 1, 1))
            self.thetak_init = Parameter(torch.Tensor(1, out_channels, 1, 1, 1))
            #self.rls.append(self.rl_k)
            self.thetas_init.append(self.thetak_init)
            self.poses_w.append(self.pos_k)

        if in_channels // groups > 1:
            self.w_i = Parameter(torch.Tensor(1, out_channels, 1, in_channels // groups, 1))
            self.b_i = Parameter(torch.Tensor(1, out_channels, 1, in_channels // groups, 1))
            #self.rl_i = Parameter(torch.Tensor(num_grad, out_channels, 1, 1, 1))
            self.thetai_init = Parameter(torch.Tensor(1, out_channels, 1, 1, 1))
            #self.rls.append(self.rl_i)
            self.thetas_init.append(self.thetai_init)
            self.poses_w.append(self.pos_i)

        self.reset_parameters()

    def pos_k(self, w, theta):
        # w_k.size = b_k.size = (1, c_out, 1, 1, prod(ks))
        pos = torch.sin(theta * self.w_k + self.b_k)
        # pos.size = (n, c_out, o_h * o_w, 1, prod(ks))
        w = w * pos
        return w

    def pos_i(self, w, theta):
        # w_i.size = b_i.size = (1, c_out, 1, c_in, 1)
        pos = torch.sin(theta * self.w_i + self.b_i)
        # pos.size = (n, c_out, o_h * o_w, c_in, 1)
        w = w * pos
        return w

    def one_forward(self, inpt, batch_size, length, w, thetas):
        for theta, pos_w in zip(thetas, self.poses_w):
            w = pos_w(w, theta)

        w = w.view(batch_size, self.out_channels, length, -1)
        # w.size = (n, c_out, o_h * o_w, c_in * prod(ks))
        output = torch.sum(inpt.unsqueeze(1).transpose(-2, -1) * w, dim=-1)
        # output.size = (n, c_out, o_h * o_w)
        return output

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        if self.prod_ks > 1:
            self.w_k.data.fill_(1.)#.uniform_(-stdv, stdv)
            self.b_k.data.uniform_(-math.pi, math.pi)
            #self.rl_k.data.fill_(1. / self.num_grad)
            self.thetak_init.data.uniform_(-stdv, stdv)

        if self.in_channels // self.groups > 1:
            self.w_i.data.fill_(1.)#.uniform_(-stdv, stdv)
            self.b_i.data.uniform_(-math.pi, math.pi)
            #self.rl_i.data.fill_(1. / self.num_grad)
            self.thetai_init.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dA(_ConvNdA):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dA, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride)
        length = inpt.size(-1) # inpt.size = (n, c_in * prod(ks), o_h * o_w)
        thetas = [theta_init.expand(batch_size, -1, length, -1, -1)
                  for theta_init in self.thetas_init]
        # theta = (n, c_out, o_h * o_w, 1, 1)
        w = self.weight.view(1, self.out_channels, 1, self.in_channels, -1)
        # w.size = (1, c_out, 1, c_in, prod(ks))

        rl = 0.1
        for i in range(self.num_grad):
            out = self.one_forward(inpt, batch_size, length, w, thetas)
            gs = grad(out.sum(), thetas, create_graph=True)
            for j, g in enumerate(gs):
                thetas[j] = thetas[j] - rl * g
                #thetas[j] = thetas[j] -  0.5 ** i * rl * g
                #thetas[j] = thetas[j] - self.rls[j][i] * g
        out = self.one_forward(inpt, batch_size, length, w, thetas)
        # output.size = (n, c_out, o_h * o_w)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.view(batch_size, self.out_channels, *h_w)]
        ret.extend([theta.view(batch_size, self.out_channels, *h_w) for theta in thetas])
        return torch.cat(ret, dim=1)


class Conv2dA2(Conv2dA):
    def pos_k(self, w, theta):
        # w_k.size = b_k.size = (1, c_out, 1, 1, prod(ks))
        pos = torch.sin(theta + self.b_k) * self.w_k
        # pos.size = (n, c_out, o_h * o_w, 1, prod(ks))
        w = w + pos
        return w

    def pos_i(self, w, theta):
        # w_i.size = b_i.size = (1, c_out, 1, c_in, 1)
        pos = torch.sin(theta + self.b_i) * self.w_i
        # pos.size = (n, c_out, o_h * o_w, c_in, 1)
        w = w + pos
        return w


class Conv2dA3(Conv2dA):
    def pos_k(self, w, theta):
        # w_k.size = b_k.size = (1, c_out, 1, 1, prod(ks))
        pos = torch.sin(theta + self.b_k)
        # pos.size = (n, c_out, o_h * o_w, 1, prod(ks))
        w_k = torch.sigmoid(self.w_k)
        w = w * (1 - w_k + w_k * pos)
        return w

    def pos_i(self, w, theta):
        # w_i.size = b_i.size = (1, c_out, 1, c_in, 1)
        pos = torch.sin(theta + self.b_i)
        # pos.size = (n, c_out, o_h * o_w, c_in, 1)
        w_i = torch.sigmoid(self.w_i)
        w = w * (1 - w_i + w_i * pos)
        return w


class _ConvNdB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdB, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            1, self.prod_ks, out_channels, in_channels // groups))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.prod_ks > 1:
            #self.w_k = Parameter(torch.Tensor(1, self.prod_ks, 1, out_channels, 1))
            self.b_k = Parameter(torch.Tensor(1, self.prod_ks, 1, out_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        if self.prod_ks > 1:
            #self.w_k.data.uniform_(-stdv, stdv)
            self.b_k.data.uniform_(-stdv, stdv)
            #self.b_k.data.uniform_(-math.pi, math.pi)
            #self.thetak_init.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dB(_ConvNdB):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride)
        inpt = inpt.view(batch_size, self.in_channels, self.prod_ks, -1).permute(0, 2, 3, 1).contiguous()
        # inpt.size = (n, prod(ks), o_h * o_w, c_in)
        # self.weight.size = (1, prod(ks), c_out, c_in)
        out = inpt.matmul(self.weight.transpose(-2, -1))
        # out.size = (n, prod(ks), o_h * o_w, c_out)
        b_cs = torch.cat((torch.cos(self.b_k), torch.sin(self.b_k)), dim=-1)
        # b_cs.size = (1, prod(ks), 1, c_out, 2)
        out = torch.sum(out.unsqueeze(-1) * b_cs, dim=1)
        # out.size = (n, o_h * o_w, c_out, 2)
        a_gt_0 = out[:, :, :, 0] > 0
        b_gt_0 = out[:, :, :, 1] > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out[:, :, :, 0])
        theta[ab_eq] = torch.atan(out[:, :, :, 1][ab_eq] / (out[:, :, :, 0][ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out[:, :, :, 0][ab_nq] / (out[:, :, :, 1][ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        #out = out / torch.norm(torch.sin(theta.view(batch_size, 1, -1, self.out_channels, 1) + self.b_k), p=2, dim=1)
        out = (torch.sin(theta) * out[:, :, :, 0] + torch.cos(theta) * out[:, :, :, 1]).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)

        '''
        # testing code
        w = torch.sin(theta.view(batch_size, 1, -1, self.out_channels, 1) + self.b_k)
        # w.size = (n, prod(ks), o_h * o_w, c_out, 1)
        w = w / torch.norm(w, p=2, dim=1, keepdim=True)
        out1 = (w * self.weight.view(1, self.prod_ks, 1, self.out_channels, self.in_channels // self.groups) \
               * inpt.unsqueeze(-2)).sum(1).sum(-1).transpose(-2, -1)
        print("out1-out:", out1-out)
        '''

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([(theta / math.pi).transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class _ConvNdB2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdB2, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            1, self.prod_ks, out_channels, in_channels // groups))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.prod_ks > 1:
            self.w_k = Parameter(torch.Tensor(1, self.prod_ks, 1, out_channels, 1))
            self.b_k = Parameter(torch.Tensor(1, self.prod_ks, 1, out_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        if self.prod_ks > 1:
            self.w_k.data.uniform_(-stdv + 0.5, stdv + 0.5)
            self.b_k.data.uniform_(-math.pi, math.pi)
            #self.thetak_init.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dB2(_ConvNdB2):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dB2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride)
        inpt = inpt.view(batch_size, self.in_channels, self.prod_ks, -1).permute(0, 2, 3, 1).contiguous()
        # inpt.size = (n, prod(ks), o_h * o_w, c_in)
        # self.weight.size = (1, prod(ks), c_out, c_in)
        out = inpt.matmul(self.weight.transpose(-2, -1))
        # out.size = (n, prod(ks), o_h * o_w, c_out)
        b_cs = torch.cat((torch.cos(self.b_k), torch.sin(self.b_k)), dim=-1)
        # b_cs.size = (1, prod(ks), 1, c_out, 2)
        self.w_k.data.clamp_(0., 1.)
        # w_k.size = (1, prod(ks), 1, c_out, 1)
        b_cs = 1. - self.w_k + self.w_k * b_cs
        out = torch.sum(out.unsqueeze(-1) * b_cs, dim=1)
        # out.size = (n, o_h * o_w, c_out, 2)
        a_gt_0 = out[:, :, :, 0] > 0
        b_gt_0 = out[:, :, :, 1] > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out[:, :, :, 0])
        theta[ab_eq] = torch.atan(out[:, :, :, 1][ab_eq] / (out[:, :, :, 0][ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out[:, :, :, 0][ab_nq] / (out[:, :, :, 1][ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        out = (torch.sin(theta) * out[:, :, :, 0] + torch.cos(theta) * out[:, :, :, 1]).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)
        theta = theta / math.pi

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([theta.transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class Conv2dB3(_ConvNdB2):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dB3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride)
        inpt = inpt.view(batch_size, self.in_channels, self.prod_ks, -1).permute(0, 2, 3, 1).contiguous()
        # inpt.size = (n, prod(ks), o_h * o_w, c_in)
        # self.weight.size = (1, prod(ks), c_out, c_in)
        out = inpt.matmul(self.weight.transpose(-2, -1))
        # out.size = (n, prod(ks), o_h * o_w, c_out)
        b_cs = torch.cat((torch.cos(self.b_k), torch.sin(self.b_k)), dim=-1)
        # b_cs.size = (1, prod(ks), 1, c_out, 2)
        self.w_k.data.clamp_(0., 1.)
        w_k = Normal(self.w_k, 0.01).rsample().clamp(0., 1.) if self.training else self.w_k
        # w_k.size = (1, prod(ks), 1, c_out, 1)
        b_cs = 1. - w_k + w_k * b_cs
        out = torch.sum(out.unsqueeze(-1) * b_cs, dim=1)
        # out.size = (n, o_h * o_w, c_out, 2)
        a_gt_0 = out[:, :, :, 0] > 0
        b_gt_0 = out[:, :, :, 1] > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out[:, :, :, 0])
        theta[ab_eq] = torch.atan(out[:, :, :, 1][ab_eq] / (out[:, :, :, 0][ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out[:, :, :, 0][ab_nq] / (out[:, :, :, 1][ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        out = (torch.sin(theta) * out[:, :, :, 0] + torch.cos(theta) * out[:, :, :, 1]).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)
        theta = theta / math.pi

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([theta.transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class _ConvNdC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdC, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            1, in_channels // groups, out_channels, self.prod_ks))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if in_channels // groups > 1:
            #self.w_i = Parameter(torch.Tensor(1, in_channels // groups, 1, out_channels, 1))
            self.b_i = Parameter(torch.Tensor(1, in_channels // groups, 1, out_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        if self.in_channels // self.groups > 1:
            #self.w_i.data.uniform_(-stdv, stdv)
            self.b_i.data.uniform_(-stdv, stdv)
            #self.b_i.data.uniform_(-math.pi, math.pi)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dC(_ConvNdC):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride)
        inpt = inpt.view(batch_size, self.in_channels, self.prod_ks, -1).transpose(-2, -1).contiguous()
        # inpt.size = (n, c_in, o_h * o_w, prod(ks))
        # self.weight.size = (1, c_in, c_out, prod(ks))
        out = inpt.matmul(self.weight.transpose(-2, -1))
        # out.size = (n, c_in, o_h * o_w, c_out)
        b_cs = torch.cat((torch.cos(self.b_i), torch.sin(self.b_i)), dim=-1)
        # b_cs.size = (1, c_in, 1, c_out, 2)
        out = torch.sum(out.unsqueeze(-1) * b_cs, dim=1)
        # out.size = (n, o_h * o_w, c_out, 2)
        a_gt_0 = out[:, :, :, 0] > 0
        b_gt_0 = out[:, :, :, 1] > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out[:, :, :, 0])
        theta[ab_eq] = torch.atan(out[:, :, :, 1][ab_eq] / (out[:, :, :, 0][ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out[:, :, :, 0][ab_nq] / (out[:, :, :, 1][ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        #out = out / torch.norm(torch.sin(theta.view(batch_size, 1, -1, self.out_channels, 1) + self.b_k), p=2, dim=1)
        out = (torch.sin(theta) * out[:, :, :, 0] + torch.cos(theta) * out[:, :, :, 1]).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)


        '''
        # testing code
        w = torch.sin(theta.view(batch_size, 1, -1, self.out_channels, 1) + self.b_i)
        # w.size = (n, c_in, o_h * o_w, c_out, 1)
        #w = w / torch.norm(w, p=2, dim=1, keepdim=True)
        out1 = (w * self.weight.view(1, self.in_channels // self.groups, 1, self.out_channels, self.prod_ks) \
               * inpt.unsqueeze(-2)).sum(1).sum(-1).transpose(-2, -1)
        print("out1-out:", out1-out)
        '''


        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([(theta / math.pi).transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class _ConvNdD(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdD, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups * self.prod_ks))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dD(_ConvNdD):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dD, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride).transpose(-2, -1)
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        # self.weight.size = (c_out, c_in * prod(ks))
        out_cos = inpt.matmul(torch.cos(self.weight).transpose(-2, -1))
        # out_cos.size = (n, o_h * o_w, c_out)
        out_sin = inpt.matmul(torch.sin(self.weight).transpose(-2, -1))
        # out_sin.size = (n, o_h * o_w, c_out)
        a_gt_0 = out_cos > 0
        b_gt_0 = out_sin > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out_cos)
        theta[ab_eq] = torch.atan(out_sin[ab_eq] / (out_cos[ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out_cos[ab_nq] / (out_sin[ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        out = (torch.sin(theta) * out_cos + torch.cos(theta) * out_sin).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)

        '''
        # testing code
        w = torch.sin(theta.view(batch_size, -1, self.out_channels, 1) + self.weight.view(1, 1, self.out_channels, -1))
        # w.size = (n, o_h * o_w, c_out, c_in * prod(ks))
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        out1 = (w * inpt.unsqueeze(-2)).sum(-1).transpose(-2, -1)
        print("out1-out:", out1-out)
        '''

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([(theta / math.pi).transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class _ConvNdD2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdD2, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            1, 1, out_channels, in_channels // groups * self.prod_ks))
        self.weight1 = Parameter(torch.Tensor(
            1, 1, out_channels, in_channels // groups * self.prod_ks))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dD2(_ConvNdD2):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dD2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride).transpose(-2, -1)
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        # self.weight1.size = (1, 1, c_out, c_in * prod(ks))
        inpt = inpt.unsqueeze(-2) * self.weight1
        # inpt.size = (n, o_h * o_w, c_out, c_in * prod(ks))
        # self.weight.size = (1, 1, c_out, c_in * prod(ks))
        out_cos = torch.sum(inpt * torch.cos(self.weight), dim=-1)
        # out_cos.size = (n, o_h * o_w, c_out)
        out_sin = torch.sum(inpt * torch.sin(self.weight), dim=-1)
        # out_sin.size = (n, o_h * o_w, c_out)
        a_gt_0 = out_cos > 0
        b_gt_0 = out_sin > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out_cos)
        theta[ab_eq] = torch.atan(out_sin[ab_eq] / (out_cos[ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out_cos[ab_nq] / (out_sin[ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        out = (torch.sin(theta) * out_cos + torch.cos(theta) * out_sin).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)

        '''
        # testing code
        w = torch.sin(theta.view(batch_size, -1, self.out_channels, 1) + self.weight.view(1, 1, self.out_channels, -1))
        # w.size = (n, o_h * o_w, c_out, c_in * prod(ks))
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        out1 = (w * inpt).sum(-1).transpose(-2, -1)
        print("out1-out:", out1-out)
        '''

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([(theta / math.pi).transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class _ConvNdD3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, num_grad=3):
        super(_ConvNdD3, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.num_grad = num_grad

        self.kernel_tensor = torch.LongTensor(self.kernel_size)
        self.stride_tensor = torch.LongTensor(self.stride)
        self.padding_tensor = torch.LongTensor(self.padding)
        self.dilation_tensor = torch.LongTensor(self.dilation)

        self.prod_ks = np.prod(kernel_size)
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups * self.prod_ks))
        self.weight1 = Parameter(torch.Tensor(
            out_channels, in_channels // groups * self.prod_ks))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dD3(_ConvNdD3):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, num_grad=3):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dD3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            _pair(0), groups, bias, num_grad)

    def forward(self, inpt):
        inpt_size = inpt.size()
        batch_size = inpt_size[0]
        h_w = torch.LongTensor(tuple(inpt_size[-2:]))
        h_w = (h_w + 2 * self.padding_tensor - self.dilation_tensor
               * (self.kernel_tensor - 1) - 1) / self.stride_tensor + 1

        inpt = F.unfold(inpt, self.kernel_size, self.dilation, self.padding, self.stride).transpose(-2, -1)
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        # self.weight.size = (c_out, c_in * prod(ks))
        out_cos = inpt.matmul((torch.cos(self.weight) * self.weight1).transpose(-2, -1))
        # out_cos.size = (n, o_h * o_w, c_out)
        out_sin = inpt.matmul((torch.sin(self.weight) * self.weight1).transpose(-2, -1))
        # out_sin.size = (n, o_h * o_w, c_out)
        a_gt_0 = out_cos > 0
        b_gt_0 = out_sin > 0
        ab_eq = (a_gt_0 == b_gt_0)
        ab_nq = (a_gt_0 != b_gt_0)
        eps = 1e-5
        theta = torch.zeros_like(out_cos)
        theta[ab_eq] = torch.atan(out_sin[ab_eq] / (out_cos[ab_eq] + eps))
        theta[ab_nq] = - torch.atan(out_cos[ab_nq] / (out_sin[ab_nq] + eps))
        # theta.size = (n, o_h * o_w, c_out)
        peak = torch.zeros_like(theta)
        peak[ab_eq * a_gt_0] = math.pi / 2.
        peak[ab_eq * (1 - a_gt_0)] = - math.pi / 2.
        peak[ab_nq * (1 - b_gt_0)] = - math.pi
        theta = peak - theta
        out = (torch.sin(theta) * out_cos + torch.cos(theta) * out_sin).transpose(-2, -1)
        # output.size = (n, c_out, o_h * o_w)

        '''
        # testing code
        w = torch.sin(theta.view(batch_size, -1, self.out_channels, 1) + self.weight.view(1, 1, self.out_channels, -1))
        # w.size = (n, o_h * o_w, c_out, c_in * prod(ks))
        # inpt.size = (n, o_h * o_w, c_in * prod(ks))
        out1 = (w * inpt.unsqueeze(-2)).sum(-1).transpose(-2, -1)
        print("out1-out:", out1-out)
        '''

        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        ret = [out.reshape(batch_size, self.out_channels, *h_w)]
        ret.extend([(theta / math.pi).transpose(-2, -1).reshape(batch_size, self.out_channels, *h_w)])
        return torch.cat(ret, dim=1)


class MaxPoolP2d(nn.MaxPool2d):
    def __init__(self, in_channels, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPoolP2d, self).__init__(kernel_size, stride, padding, dilation,
                                         return_indices, ceil_mode)
        self.in_channels = in_channels

    def forward(self, inpts):
        inpts = torch.split(inpts, self.in_channels, dim=1)
        out, where = F.max_pool2d(inpts[0], self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode, True)
        out_size = out.size()
        where = where.view(out_size[0], out_size[1], -1)
        ret = [out]
        for inpt in inpts[1:]:
            inpt = torch.gather(inpt.view(out_size[0], out_size[1], -1), -1, where).view_as(out)
            ret.append(inpt)
        ret = torch.cat(ret, dim=1)
        if self.return_indices:
            return ret, where.view_as(out)
        else:
            return ret


class NonLiPA(nn.Module):
    def __init__(self, in_channels, nonlinear):
        super(NonLiPA, self).__init__()
        self.in_channels = in_channels
        self.nonlinear = nonlinear

    def forward(self, inpts):
        inpt = inpts[:, :self.in_channels]
        inpt = self.nonlinear(inpt)
        inpts = torch.tanh(inpts[:, self.in_channels:])
        return torch.cat((inpt, inpts), dim=1)

    def __repr__(self):
        return self.nonlinear.__class__.__name__ + 'PA(' +  self.nonlinear.extra_repr() + ')'


class NonLiPB(nn.Module):
    def __init__(self, in_channels, nonlinear):
        super(NonLiPB, self).__init__()
        self.in_channels = in_channels
        self.nonlinear = nonlinear

    def forward(self, inpts):
        inpt = inpts[:, :self.in_channels]
        inpt = self.nonlinear(inpt)
        inpts = inpts[:, self.in_channels:]
        return torch.cat((inpt, inpts), dim=1)

    def __repr__(self):
        return self.nonlinear.__class__.__name__ + 'PB(' +  self.nonlinear.extra_repr() + ')'


class Cat(nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim =dim

    def forward(self, inpt):
        return torch.cat(inpt, dim=self.dim)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    x = torch.randn(2, 2, 8, 8).to(device)
    convp = Conv2dD2(2, 3, 2).to(device)
    out = convp(x)
    out.sum().backward()
    # for p in convp.parameters():
    #     print(p.grad)
    print(out.size())

    # conv = nn.Conv2d(2, 3, 2).to(device)
    # conv.weight = convp.weight
    # conv.bias = convp.bias
    # print(conv(x))

    maxpool = MaxPoolP2d(in_channels=3, kernel_size=2, stride=2, return_indices=True)

    print(out)
    out = maxpool(out)
    print(out[1])
    print(out[0][1])

'''
class ModuleCNNP(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(ModuleCNNP, self).__init__()
        self.conv1 = ConvP2d(in_channels, hid_channels, kernel_size=3),
        self.bn1 = nn.BatchNorm2d(hid_channels),
        self.mp = MaxPoolP2d(kernel_size=2, stride=2),
        self.conv2 = nn.Conv2d(hid_channels, hid_channels * 2, kernel_size=3),
        self.bn2 = nn.BatchNorm2d(hid_channels),
        self.nonl2 = nn.ReLU(inplace=True),
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2),
        self.linear1 = nn.Linear(50 * 5 * 5, 1024),
        self.linear2 = nn.Linear(1024, 128),
        self.linear3 = nn.Linear(128, 10)
'''