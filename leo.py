from torch.autograd import grad
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from collections import OrderedDict
import copy
import os

from utils import *
from fuzzymeta import clip_grad_norm


from torch.distributions import Normal



def _to_parall(m):
    if type(m) == nn.Linear:
        return LinearParall()
    if type(m) == nn.Conv2d:
        return Conv2dParall()
    if type(m) == nn.ConvTranspose2d:
        return Conv2dTParall()
    if type(m) == nn.LayerNorm:
        return LayerNormParall()
    if type(m) == nn.BatchNorm1d or \
        type(m) == nn.BatchNorm2d or \
        type(m) == nn.BatchNorm3d:
        return BatchNormParall()
    if type(m) == nn.InstanceNorm1d or \
        type(m) == nn.InstanceNorm2d or \
        type(m) == nn.InstanceNorm3d:
        return InstanceNormParall()
    if type(m) == nn.Dropout or \
        type(m) == nn.Dropout2d or \
        type(m) == nn.Dropout3d:
        return DropoutParall()
    if type(m) == Capsule:
        return CapsuleParall()
    if type(m) == CapsuleShare:
        return CapsuleSParall()
    #if type(m) == CapsuleConv2d:
        #return CapsuleC2dParall()
    if type(m) == CapsuleCo:
        return CapsuleCoParall()
    if type(m) == nn.AvgPool1d:
        return AvgPool1dParall()
    if type(m) == nn.AvgPool2d:
        return AvgPool2dParall()
    if type(m) == nn.MaxPool2d:
        return MaxPool2dParall()

    if type(m) == LEOConv2d:
        return LEOConv2dParall()
    if type(m) == LEOBlock2DOne:
        return LEOBlock2DParall(m)

    return sameparall(m)


class _MetaBase(nn.Module):
    def __init__(self):
        super(_MetaBase, self).__init__()
        self.nonfinetune = None
        self.lam = None

    def get_lossf(self, lossf):
        self.lossf = lossf

    def init_inner(self, num):
        pass

    def create_weights(self, num):
        pass

    def inner_fit(self, feat, label):
        self.feat_size = feat.size()
        self.num = self.feat_size[0]
        self.nkshot = self.feat_size[1]
        feat = feat.transpose(0, 1).contiguous()
        label_size = list(label.size())
        label_size = [label_size[0] * label_size[1]] + label_size[2:]
        self.label = label.transpose(0, 1).contiguous()
        label = self.label.view(*label_size)
        self.feat = self.abstract(feat)
        feat = self.feat

        kl = self.init_inner(self.num)
        self.loss_p = self.beta * kl
        for i in range(self.inner_num):
            self.create_weights(self.num)
            output = self.net_C[0](feat)
            if i == 0:
                output_size = list(output.size())
                output_size = [output_size[0] * output_size[1]] + output_size[2:]
            loss = self.num * self.lossf(output.view(*output_size), label)

            grads = grad(loss, self.use_wlist, create_graph=self.training)
            self.inner_update(grads)

        self.loss_p = self.loss_p + self.gamma * self.get_stopg_loss()
        # corr_loss = 0
        # for i, m_C in enumerate(self.leolayers_C.values(), 0):
        #     corr_loss = corr_loss + m_C.decode.get_corr_loss()
        # self.loss_p = self.loss_p + corr_loss
        return loss

    def get_stopg_loss(self):
        return 0.

    def inner_update(self, grads):
        pass

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(feat)
            feat = feat if self.training else feat.detach()
        return feat

    def inner_test(self, feat, label, classify=False, reduce=True, gagree=False):
        self.feat_size = feat.size()
        self.num = self.feat_size[0]
        self.nkquery = self.feat_size[1]
        feat = feat.transpose(0, 1).contiguous()
        self.label_size = list(label.size())
        self.label_size = [self.label_size[0] * self.label_size[1]] + self.label_size[2:]
        self.label = label.transpose(0, 1).contiguous()
        label = self.label.view(*self.label_size)
        feat = self.abstract(feat)
        self.create_weights(self.num)
        self.net_C[0].inner_eval()

        output = self.net_C[0](feat)
        output_size = list(output.size())
        output_size = [output_size[0] * output_size[1]] + output_size[2:]
        if reduce:
            return self.num * self.lossf(output.view(*output_size), label),\
                   self.loss_p
        else:
            loss = self.lossf(output.view(*output_size), label, reduction='none')
            loss = loss.view(-1, self.num)
            if gagree:
                return loss.mean(0), self.loss_p
            elif classify:
                self.label = self.label.view(self.label_size[0] // self.num, self.num)
                c = (torch.argmax(output, 2) == self.label).type(FloatTensor)
                return loss.mean(0), c.mean(0)
            else:
                return loss.mean(0)

    def forward(self, feata, labela, featb):
        self.inner_fit(feata, labela)
        feat_size = featb.size()
        self.num = feat_size[0]
        self.nkquery = feat_size[1]
        featb = featb.transpose(0, 1).contiguous()
        if self.nonfinetune is not None:
            featb = self.nonfinetune(featb)
            featb = featb if self.training else featb.detach()
        self.create_weights(self.num)
        self.net_C[0].inner_eval()
        return self.net_C[0](featb).transpose(0, 1).contiguous()

    def train(self, mode=True):
        self.net_C[0].train(mode)
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


class LEO(_MetaBase):
    def __init__(self, inner_num, inner_lr, layers, beta=0.1, gamma=1e-8):
        super(LEO, self).__init__()
        if isinstance(layers[0], NonFineTune):
            self.nonfinetune = layers[0]
            self.net = nn.Sequential(*layers[1:])
        else:
            self.net = nn.Sequential(*layers)
        self.beta = beta
        self.gamma = gamma
        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.net_C = [SequentialParall()]
        self.wnames = []
        self.mdict = []
        self.wnames_leo = []
        self.mdict_leo = []

        self.leolayers = OrderedDict()
        self.leolayers_C = OrderedDict()
        self.bws = []

        for i, module in enumerate(self.net._modules.values()):
            if isinstance(module, LEOBase):
                self.leolayers[str(i)] = module

            if isinstance(module, LEOBlock):
                j = 0
                for m in module._modules.values():
                    if isinstance(m, LEOBase):
                        self.leolayers[str(i)+str(j)] = m
                    j += 1

            moduleparall = _to_parall(module)
            self.net_C[0].add_module(str(i), moduleparall)

            if isinstance(module, LEOBase):
                self.leolayers_C[str(i)] = moduleparall
                self.mdict_leo.append(moduleparall.__dict__)
                self.wnames_leo.append('latents')

            if isinstance(module, LEOBlock):
                j = 0
                for m, mP in zip(module._modules.values(), moduleparall._modules.values()):
                    if isinstance(m, LEOBase):
                        self.leolayers_C[str(i) + str(j)] = mP
                        self.mdict_leo.append(mP.__dict__)
                        self.wnames_leo.append('latents')
                    j += 1

                    # if isinstance(m, FCNN2D) and hasattr(m, 'bw'):
                    #     if isinstance(m.bw, BatchWeight):
                    #         self.bws.append(m.bw)
                    #     continue

                    for name, param in m.named_parameters(recurse=not isinstance(m, LEOBase)):
                        self.mdict.append(mP.__dict__)
                        self.wnames.append(name)
                continue

            # if isinstance(module, FCNN2D) and hasattr(module, 'bw'):
            #     if isinstance(module.bw, BatchWeight):
            #         self.bws.append(module.bw)
            #     continue

            if isinstance(module, NFTModule):
                continue

            for name, param in module.named_parameters(recurse=not isinstance(module, LEOBase)):
                self.mdict.append(moduleparall.__dict__)
                self.wnames.append(name)

        self.wnames.extend(self.wnames_leo)
        self.mdict.extend(self.mdict_leo)
        self.len_leo = len(self.wnames_leo)
        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * self.len_w
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)

        self.bws = tuple(self.bws)

    def init_inner(self, num):
        i = 0
        for m, m_C in zip(self.net._modules.values(), self.net_C[0]._modules.values()):
            m_C.get_parameters(m, num)
            if isinstance(m, LEOBlock):
                for m_s,m_C_s in zip(m._modules.values(), m_C._modules.values()):
                    recurse = not isinstance(m_s, LEOBase)
                    for name, _ in m_s.named_parameters(recurse=recurse):
                        self.use_wlist[i] = getattr(m_C_s, name)
                        i += 1
                continue
            if isinstance(m, NFTModule):
                continue
            recurse = not isinstance(m, LEOBase)
            for name, _ in m.named_parameters(recurse=recurse):
                self.use_wlist[i] = getattr(m_C, name)
                i += 1

        self.net_C[0].inner_train()

        self.latents_list = []
        kl = 0.
        for i, m_C in enumerate(self.leolayers_C.values(), 0):
            latents, _kl = m_C.create_latents(self.feat)
            self.latents_list.append(latents)
            self.use_wlist[i - self.len_leo] = latents
            kl = kl + _kl

        for bw in self.bws:
            bw.reset(num)

        return kl

    def get_stopg_loss(self):
        if self.training:
            loss = 0.
            for ls_start, ls in zip(self.latents_list, self.use_wlist[-self.len_leo:]):
                loss = loss + F.mse_loss(ls_start, ls)
            return loss
        else:
            return 0.

    def create_weights(self, num):
        pass

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - self.inner_lr * g
            self.use_wlist[i] = m[wn]

    def forward(self, feata, labela, featb):
        self.inner_fit(feata, labela)
        feat_size = featb.size()
        self.num = feat_size[0]
        self.nkquery = feat_size[1]
        featb = featb.transpose(0, 1).contiguous()
        if self.nonfinetune is not None:
            featb = self.nonfinetune(featb)
            featb = featb if self.training else featb.detach()
        self.create_weights(self.num)
        self.net_C[0].inner_eval()
        return self.net_C[0](featb).transpose(0, 1).contiguous()

    def extra_repr(self):
        s = ('inner_lr={inner_lr}, inner_num={inner_num}, beta={beta}, gamma={gamma}')
        return s.format(**self.__dict__)


class LEOWeD(LEO):
    def __init__(self, inner_num, inner_lr, inner_wd, layers, beta=0.1, gamma=1e-8):
        super(LEOWeD, self).__init__(inner_num, inner_lr, layers, beta, gamma)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]

    def extra_repr(self):
        s = ('inner_lr={inner_lr}, inner_num={inner_num}, inner_wd={inner_wd}, beta={beta}, gamma={gamma}')
        return s.format(**self.__dict__)


class LEOCGN(LEO):
    def __init__(self, inner_num, inner_lr, inner_wd, layers, beta=0.1, gamma=1e-8):
        super(LEOCGN, self).__init__(inner_num, inner_lr, layers, beta, gamma)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)

    def inner_update(self, grads):
        clip_grad_norm(grads, max_norm=1.)

        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]

    def extra_repr(self):
        s = ('inner_lr={inner_lr}, inner_num={inner_num}, inner_wd={inner_wd}, beta={beta}, gamma={gamma}')
        return s.format(**self.__dict__)


class LEOEncode(NFTSequential):
    def forward(self, x, drop_rate):
        in_size = list(x.size())
        re_size = [in_size[0] * in_size[1]] + in_size[2:]
        x = x.view(*re_size)
        x = F.dropout(x, p=drop_rate, training=self.training)
        for module in self._modules.values():
            x = module(x)
        out_size = list(x.size())
        re_size = [in_size[0], in_size[1]] + out_size[1:]
        return x.view(*re_size).mean(dim=0).view(in_size[1], 2, -1)


class LEODecode(NFTModule):
    def __init__(self, latents_size, weight_size, dete=False):
        super(LEODecode, self).__init__()
        self.latents_size = latents_size
        self.weight_size = weight_size
        self.dete = dete
        ws_total = 1
        for n in weight_size: ws_total *= n
        self.ws_total = ws_total
        self.stddev_offset = np.sqrt(2. / (latents_size + ws_total))
        self.linear = nn.Linear(latents_size, ws_total) if dete else \
            nn.Linear(latents_size, ws_total * 2)

    def forward(self, x):
        if self.dete:
            return self.linear(x).view(-1, *self.weight_size)
        else:
            distp = self.linear(x).view(-1, 2, self.ws_total)
            means = distp[:, 0]
            stddev = torch.exp(distp[:, 1]) - (1. - self.stddev_offset)
            stddev = stddev.clamp_min(1e-10)
            normal = Normal(loc=means, scale=stddev)
            weight = normal.rsample()
            return weight.view(-1, *self.weight_size)

    def get_corr_loss(self):
        corr = _corr(self.linear.weight)
        return F.l1_loss(corr, torch.eye(len(corr)))

    def extra_repr(self):
        s = ('dete={dete}')
        return s.format(**self.__dict__)


def _cov(x):
    assert len(x.size()) == 2, 'The dimension of x is unequal to 2!'
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.mm(x, x.t()) / (len(x) - 1)
    return cov

def _corr(x):
    y = x.std(dim=1, keepdim=True)
    y = torch.mm(y, y.t())
    return _cov(x) / y


class LEOBase(nn.Module):
    def __init__(self, encode, latents_size, weight_size, drop_rate=0.):
        super(LEOBase, self).__init__()
        self.encode = LEOEncode(*encode)
        self.decode = LEODecode(latents_size, weight_size)
        self.drop_rate = drop_rate
        self.inner_training = False

    def extra_repr(self):
        s = ('drop_rate={drop_rate}')
        return s.format(**self.__dict__)


class LEOConv2d(LEOBase):
    def __init__(self, encode, latents_size,
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, drop_rate=0.):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        weight_size = torch.Size((out_channels, in_channels, *self.kernel_size))
        super(LEOConv2d, self).__init__(encode, latents_size, weight_size, drop_rate)

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
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ', drop_rate={drop_rate}'
        return s.format(**self.__dict__)


class LEOParall(ModuleParall):
    def __init__(self):
        super(LEOParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.encode = model.encode
        self.decode = model.decode
        self.drop_rate = model.drop_rate
        self.training = model.training
        self.inner_training = model.inner_training
        self.num = num

    def create_latents(self, feat):
        distp = self.encode(feat, self.drop_rate)
        if self.training:
            means = distp[:, 0]
            stddev = torch.exp(distp[:, 1])
            stddev = stddev.clamp_min(1e-10)
            normal = Normal(loc=means, scale=stddev)
            self.latents = normal.rsample()
            kl = torch.mean(normal.log_prob(self.latents) -
                            Normal(loc=0., scale=1.).log_prob(self.latents))
        else:
            self.latents = distp[:, 0]
            kl = 0.
        return self.latents, kl

    def forward(self, x):
        weight = self.decode(self.latents)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LEOConv2dParall(LEOParall):
    def __init__(self):
        super(LEOConv2dParall, self).__init__()

    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)
        self.stride = model.stride
        self.padding = model.padding
        self.dilation = model.dilation
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.kernel_size = model.kernel_size

        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias_r(self.bias)

    def forward(self, x):
        weight = self.decode(self.latents).view(
            self.num * self.out_channels, self.in_channels, *self.kernel_size)

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])

    def bias_r(self, bias):
        return bias.repeat(self.num)

    def repeat_lr(self, lr, wname):
        lr = lr.view(self.num, 1).repeat(1, self.in_channels)
        if wname == 'bias':
            lr = lr.view(-1)
        return lr


class LEOBlock(nn.Module):
    pass


class LEOBlock2DOne(LEOBlock):
    expansion = 1

    def __init__(self, embedding_size, latents_size, inplanes, planes, stride=1, drop_rate=0., downsample=None):
        super(LEOBlock2DOne, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()

        self.conv2 = LEOConv2d(
            encode=(
                nn.Conv2d(in_channels=embedding_size, out_channels=latents_size * 2,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(latents_size * 2),
                nn.ReLU(),
                nn.Conv2d(in_channels=latents_size * 2, out_channels=latents_size * 2,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(latents_size * 2),
                nn.ReLU(),
                nn.Conv2d(in_channels=latents_size * 2, out_channels=latents_size * 2,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.AdaptiveAvgPool2d(1)
            ),
            latents_size=latents_size,
            in_channels=planes, out_channels=planes,
            kernel_size=3, stride=1, padding=1, bias=False, drop_rate=drop_rate)

        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)

        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion, track_running_stats=False),
            )
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


class LEOBlock2DParall(ModuleParall):
    expansion = 1
    def __init__(self, m):
        super(LEOBlock2DParall, self).__init__()
        self.conv1 = _to_parall(m.conv1)
        self.bn1 = BatchNormParall()
        self.relu = nn.ReLU()
        self.conv2 = _to_parall(m.conv2)

        self.bn2 = BatchNormParall()
        downsample = None
        if m.downsample is not None:
            downsample = SequentialParall(
                Conv2dParall(),
                BatchNormParall(),
            )
        self.downsample = downsample

    def get_parameters(self, model, num=1):
        self.conv1.get_parameters(model.conv1, num)
        self.bn1.get_parameters(model.bn1, num)
        self.conv2.get_parameters(model.conv2, num)
        self.bn2.get_parameters(model.bn2, num)
        if self.downsample is not None:
            self.downsample.get_parameters(model.downsample, num)

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