#!/usr/bin/python3.6

from pylab import *
import numpy as np
from tqdm import tqdm
import os
import ast
import time
import gc
import pickle
import random

from parallel import *
import torch.optim as optim
from torch.optim.lr_scheduler import *
from bisect import bisect_right

torch.cuda.set_device(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

torch.backends.cudnn.benchmark = True


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    printl(name)
    printl(model)
    printl("The number of parameters: {}".format(num_params))

resultpath = "result.txt"
def set_resultpath(path):
    global resultpath
    resultpath = path


def get_resultpath():
    global resultpath
    return resultpath


def printl(strin,end='\n'):
    with open(resultpath, "a") as log:
        log.write(str(strin)+end)
    print(strin,end=end)


class Printl(object):
    def __init__(self, path):
        self.path = path

    def __call__(self, strin,end='\n'):
        with open(self.path, "a") as log:
            log.write(str(strin) + end)
        print(strin, end=end)

    def clear(self):
        with open(self.path, "w") as log:
            log.write("")


class Scheduler(object):
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def see(self):
        self.scheduler.lr_lambdas[0].see('Before scheduler step.')
        self.optimizer.step()
        self.scheduler.step()
        self.scheduler.lr_lambdas[0].see('After scheduler step.')


def sche2str(sche):
    if isinstance(sche, MultiStepLR):
        return "MultiStepLR(milestones=" + str(sche.milestones) + \
               ", gamma=" + str(sche.gamma) + ")"
    elif isinstance(sche, LambdaLR):
        return "LambdaLR(" + str(sche.lr_lambdas[0]) + ")"
    return ""


class MSCCALR(object):
    """
    Multi-Step + Cyclic Cosine Annealing.
    """
    def __init__(self, milestones, gamma=0.1, interval=2000):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones1 = milestones[:-1]
        self.milestones2 = milestones[-1]
        self.gamma = gamma
        self.interval = interval

    def __call__(self, epoch):
        return self.forward(epoch)

    def forward(self, epoch):
        if epoch < self.milestones2:
            self.epoch = epoch
            self.rate = self.gamma ** bisect_right(self.milestones1, epoch)
            return self.rate
        else:
            self.alpha0 = self.gamma ** bisect_right(self.milestones1, epoch) / 2.
            self.forward = self.cyclic_annealing
            return self(epoch)

    def cyclic_annealing(self, epoch):
        self.epoch = epoch
        self.rate = self.alpha0 * (1 + math.cos(math.pi * ((epoch - self.milestones2) % self.interval) / self.interval))
        return self.rate

    def see(self, s):
        if -4 <= ((self.epoch - self.milestones2) % self.interval - self.interval) <= 4 or (self.epoch - self.milestones2) % self.interval == 0:
            printl(s)
            printl('Scheduler epoch=%d, rate=%.6f' % (self.epoch, self.rate))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'milestones1=' + str(self.milestones1) + ',' \
            + 'milestones2=' + str(self.milestones2) + ',' \
            + 'gamma=' + str(self.gamma) + ',' \
            + 'interval=' + str(self.interval) + ')'


class MSCCALR2(MSCCALR):
    def __init__(self, milestones, milestones2, gamma=0.1, interval=2000):
        super(MSCCALR2, self).__init__(milestones, gamma, interval)
        self.milestones1.append(milestones[-1])
        self.milestones2 = milestones2

    def forward(self, epoch):
        if epoch < self.milestones2:
            self.epoch = epoch
            self.rate = self.gamma ** bisect_right(self.milestones1, epoch)
            return self.rate
        else:
            self.alpha0 = self.gamma ** bisect_right(self.milestones1, epoch)
            return self.cyclic_annealing(epoch)


class CCALR(object):
    """
    Multi-Step + Cyclic Cosine Annealing.
    """
    def __init__(self, alpha0=1., interval=2000):
        self.alpha0 = alpha0 / 2.
        self.interval = interval
        self.milestones2 = 0

    def __call__(self, epoch):
        self.epoch = epoch
        self.rate = self.alpha0 * (1 + math.cos(
            math.pi * (self.epoch % self.interval) / self.interval))
        return self.rate

    def see(self, s):
        if -2 <= ((self.epoch - self.milestones2) % self.interval - self.interval) <= 2:
            printl(s)
            printl('Scheduler epoch=%d, rate=%.6f' % (self.epoch, self.rate))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'alpha0=' + str(self.alpha0 * 2.) + ',' \
            + 'interval=' + str(self.interval) + ')'


class View(nn.Module):

    def __init__(self,*size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        s = x.size()
        return x.view(s[0],s[1], *self.size)

    def extra_repr(self):
        return 'size={size}'.format(**self.__dict__)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0,self.dim1).contiguous()

    def extra_repr(self):
        return 'dim0={dim0}, dim1={dim1}'.format(**self.__dict__)


class ViewNonFT(nn.Module):

    def __init__(self,*size):
        super(ViewNonFT, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), *self.size)

    def extra_repr(self):
        return 'size={size}'.format(**self.__dict__)


class SeqMatch(nn.Sequential):
    def __init__(self, *args):
        super(SeqMatch, self).__init__(*args)

    def forward(self, x):
        input_size = x.size()
        if len(input_size) == 4:
            return self.call(x.view(input_size[0] * input_size[1], 1, input_size[-2], input_size[-1])
                             ).view(input_size[0], self.in_channels, 1, -1).mean(0)
        elif len(input_size) == 5:
            return self.call(x.view(input_size[0] * input_size[1] * input_size[2], 1, input_size[-2], input_size[-1])
                             ).view(input_size[0], input_size[1] * input_size[2], 1, -1).mean(0)

    def call(self, x):
        for module in self._modules.values():
            x = module(x)
        return F.hardtanh(x) / 2. + 0.5


class _Capsule(nn.Module):
    def __init__(self):
        super(_Capsule, self).__init__()

    def rout(self, u_hat, out_channels):
        c = self.c
        u = u_hat if self.retain_grad else u_hat.detach()
        for _ in range(self.routings - 1):
            v = squash(torch.sum(u * c, dim=1), dim=1)
            b = u * v.view(-1, 1, out_channels)
            c = F.softmax(b, 2)
        v = squash(torch.sum(u_hat * c, dim=1), dim=1)
        return v.unsqueeze(1) / 2. + 0.5


class CapsuleMatch(_Capsule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, routings=3, retain_grad=False):
        super(CapsuleMatch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.routings = routings
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, self.conv.out_channels)).type(FloatTensor)
        self.c = F.softmax(b, 2)

    def forward(self, x):
        input_size = x.size()
        u_hat = self.conv(x.view(input_size[0] * input_size[1] * input_size[2], self.conv.in_channels, input_size[-2], input_size[-1])
                          ).view(input_size[0], input_size[1], input_size[2], self.conv.out_channels, -1
                                 ).permute(1,2,0,4,3).reshape(input_size[1] * input_size[2], -1, self.conv.out_channels)
        return self.rout(u_hat, self.conv.out_channels)


class CapPred(nn.Sequential):
    def __init__(self, *args):
        super(CapPred, self).__init__(*args)

    def forward(self, x):
        input_size = x.size()
        x = self.call(x.view(input_size[0] * input_size[1] * input_size[2], 1, input_size[-2], input_size[-1]))
        output_size = x.size()
        return x.view(input_size[0], input_size[1], input_size[2], output_size[-3], output_size[-2], output_size[-1])

    def call(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class CapChMatch(_Capsule):
    def __init__(self, in_channels, set_num, kernel_size,
                 routings=3, retain_grad=False):
        super(CapChMatch, self).__init__()
        self.in_channels = in_channels
        self.set_num = set_num
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else kernel_size
        self.k_len = self.kernel_size[0] * self.kernel_size[1]
        self.routings = routings
        self.retain_grad = retain_grad
        b = Variable(torch.zeros(1, 1, set_num)).type(FloatTensor)
        self.c = F.softmax(b, 2)

        self.weight = Parameter(torch.Tensor(1, 1, self.in_channels, self.k_len, 1, set_num))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels * self.k_len)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, indexm, padding):
        input_size = x.size()

        x = F.pad(x, padding, mode='constant', value=0)
        u = torch.index_select(x.view(input_size[0], input_size[1], self.in_channels, -1), -1, indexm
                               ).view(input_size[0], input_size[1], self.in_channels, self.k_len, -1, 1) * self.weight
        u_hat = u.permute(1,3,0,2,4,5).reshape(
            input_size[1] * self.k_len, -1, self.set_num)

        '''
        u_hat = torch.sum(u.view(input_size[0], input_size[1], self.in_channels,
                       self.k_len, -1, self.set_num),4).permute(1, 3, 0, 2, 4).reshape(
            input_size[1] * self.k_len, -1, self.set_num)
        '''
        return self.rout(u_hat, self.set_num)

    def extra_repr(self):
        s = ('in_channels={in_channels}, set_num={set_num}, '
             'routings={routings}, retain_grad={retain_grad}')
        return s.format(**self.__dict__)


class CapChMatchShell(nn.Module):
    def __init__(self, match, kernel_size, stride=1, padding=0, dilation=1):
        super(CapChMatchShell, self).__init__()
        self.match = match
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)

    def forward(self, x):
        #print(x.size())
        inp_size = x.size()[-2:]
        row_size = inp_size[0] + 2 * self.padding[0]
        column_size = inp_size[1] + 2 * self.padding[1]
        im_index = torch.arange(0, row_size * column_size).view(row_size, column_size)
        index0 = torch.arange(0, inp_size[0] + 2 * self.padding[0] -
                              self.dilation[0] * (self.kernel_size[0] - 1), self.stride[0])
        index1 = torch.arange(0, inp_size[1] + 2 * self.padding[1] -
                              self.dilation[1] * (self.kernel_size[1] - 1), self.stride[1])
        indexm0 = torch.index_select(torch.index_select(im_index, 0, index0), 1, index1).view(-1)
        indexm = [(indexm0 + self.dilation[1] * i + self.dilation[0] * j * column_size)
                  for j in range(self.kernel_size[0]) for i in range(self.kernel_size[1])]
        self.indexm = torch.cat(indexm).type(LongTensor)
        self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.forward = self.call
        return self.match(x, self.indexm, self.padding)

    def call(self, x):
        return self.match(x, self.indexm, self.padding)


# class NoMatch(nn.Module):
#     def __init__(self):
#         super(NoMatch, self).__init__()
#
#     def forward(self, site):
#         return torch.sigmoid(self.weight[site[0]:site[1]])


class EachMatch(nn.Module):
    def __init__(self, set_num):
        super(EachMatch, self).__init__()
        self.set_num = set_num
        self.start = 0
        self.end = 0

    def add_field(self, num):
        self.start = self.end
        self.end += num
        return self.start, self.end

    def init_params(self):
        self.weight = Parameter(torch.Tensor(self.end, 1, self.set_num))
        stdv = 1. / np.sqrt(self.set_num)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, site):
        return torch.sigmoid(self.weight[site[0]:site[1]])


def margin_loss(inpt, target, m_plus=0.9, m_minus=0.1,lambd=0.5, reduction='elementwise_mean'):
    byte = torch.ones_like(inpt).type(ByteTensor)
    batch_size = target.size(0)
    byte[torch.arange(0, batch_size).type(LongTensor), target] = 0
    if reduction == 'elementwise_mean':
        return (torch.sum(F.relu(m_plus - inpt.gather(1, target.unsqueeze(1))) ** 2)\
               + lambd * torch.sum(F.relu(inpt[byte] - m_minus) ** 2)) / batch_size
    elif reduction == 'sum':
        return (torch.sum(F.relu(m_plus - inpt.gather(1, target.unsqueeze(1))) ** 2) \
                + lambd * torch.sum(F.relu(inpt[byte] - m_minus) ** 2))
    elif reduction == 'none':
        return (F.relu(m_plus - inpt.gather(1, target.unsqueeze(1))) ** 2)[:,0] + \
               lambd * torch.sum(F.relu(inpt[byte] - m_minus).view(batch_size, -1) ** 2, dim=1)


def margin_l1_loss(inpt, target, m_plus=0.9, m_minus=0.1,lambd=0.5, reduction='elementwise_mean'):
    byte = torch.ones_like(inpt).type(ByteTensor)
    batch_size = target.size(0)
    byte[torch.arange(0, batch_size).type(LongTensor),target] = 0
    if reduction == 'elementwise_mean':
        return (torch.sum(F.relu(m_plus - inpt.gather(1, target.unsqueeze(1))))\
               + lambd * torch.sum(F.relu(inpt[byte] - m_minus))) / batch_size
    elif reduction == 'sum':
        return (torch.sum(F.relu(m_plus - inpt.gather(1, target.unsqueeze(1)))) \
                + lambd * torch.sum(F.relu(inpt[byte] - m_minus)))
    elif reduction == 'none':
        return (F.relu(m_plus - inpt.gather(1, target.unsqueeze(1))))[:,0] + \
               lambd * torch.sum(F.relu(inpt[byte] - m_minus).view(batch_size, -1), dim=1)


def multi_margin_loss(inpt, target, p=1, margin=1, reduction='elementwise_mean'):
    byte = torch.ones_like(inpt).type(ByteTensor)
    batch_size = inpt.size(0)
    class_num = inpt.size(1)
    byte[torch.arange(0, batch_size).type(LongTensor),target] = 0
    n_inpt = inpt[byte].view(batch_size, class_num - 1)
    if reduction == 'elementwise_mean':
        return torch.sum(F.relu(margin - inpt.gather(1, target.unsqueeze(1)) + n_inpt) ** p
                         ) / (class_num * batch_size)
    elif reduction == 'sum':
        return torch.sum(F.relu(margin - inpt.gather(1, target.unsqueeze(1)) + n_inpt) ** p
                         ) / class_num
    elif reduction == 'none':
        return torch.sum(F.relu(margin - inpt.gather(1, target.unsqueeze(1)) + n_inpt) ** p
                         , dim=1) / class_num

'''
if __name__ == '__main__':
    target = torch.arange(0,9)
    target[-1] = 7
    y = torch.randn(9,8)
    print(multi_margin_loss(y, target, reduction = 'none'))
    print(F.multi_margin_loss(y, target, reduction = 'none'))
'''

def mean_cross_entropy(inpt, target, reduce=True):
    if reduce:
        batch_size = target.size(0)
        return (- torch.sum(torch.log(inpt.gather(1, target.unsqueeze(1)))) + \
               torch.sum(torch.log(torch.sum(inpt, dim=1)))) / batch_size
    else:
        return - torch.log(inpt.gather(1, target.unsqueeze(1)))[:, 0] + \
               torch.log(torch.sum(inpt, dim=1))


def plt_result(path, iter=1, y_valu='Loss', y_lim=None, picshow=True):
    if y_valu == 'Loss':
        lossa_train = np.load(path + '/lossa_train.npy')
        lossa_val = np.load(path + '/lossa_val.npy')
        lossb_train = np.load(path + '/lossb_train.npy')
        lossb_val = np.load(path + '/lossb_val.npy')
        iter_best = np.argmin(lossb_val)
        printl('Iter%d is the best lossb_val: %.6f' % (iter_best * iter, lossb_val[iter_best]))
    elif y_valu == 'Accuracy':
        lossa_train = np.load(path + '/accua_train.npy')
        lossa_val = np.load(path + '/accua_val.npy')
        lossb_train = np.load(path + '/accub_train.npy')
        lossb_val = np.load(path + '/accub_val.npy')
        iter_best = np.argmax(lossb_val)
        printl('Iter%d is the best accub_val: %.6f' % (iter_best * iter, lossb_val[iter_best]))

    len_step = (len(lossa_train) - 1) * iter
    x = np.arange(0, len_step + iter, iter)

    fig = plt.figure(num=1, figsize=(5, 4))
    plt.style.use('seaborn_grayback')

    fig.tight_layout()
    plt.xlim(0, len_step)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.plot(x, lossa_train, label='train', color='firebrick', linestyle='-')
    plt.plot(x, lossa_val, label='val', color='forestgreen', linestyle='-')
    plt.xlabel('Iter')
    plt.ylabel(y_valu + ' on shots')
    plt.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path + '/curve_on_shots.png', dpi=800)
    plt.pause(0.01)
    fig.clear()

    fig = plt.figure(num=1, figsize=(5, 4))
    plt.style.use('seaborn_grayback')
    fig.tight_layout()
    plt.xlim(0, len_step)
    plt.plot(x, lossb_train, label='train', color='firebrick', linestyle='--')
    plt.plot(x, lossb_val, label='val', color='forestgreen', linestyle='--')
    plt.xlabel('Iter')
    plt.ylabel(y_valu + ' on query')
    plt.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path + '/curve_on_queries.png', dpi=800)
    plt.pause(0.01)
    if picshow:
        plt.show()
    else:
        fig.clear()

'''
if __name__ == '__main__':
    path = '../../Result/mini/fuzzymeta/5way_5shot_15query'
    plt_result(path, iter=200, y_valu='Loss')
'''

def module_test(get_module, testset, on_set, config, classify=True):
    dataset = torch.utils.data.DataLoader(testset, batch_size=config['meta_batch_size'],
                                          shuffle=False, num_workers=8)

    model = '_i' + str(config['test_iter']) + '.pkl'
    load_path = os.path.join(config['save_path'], model)
    metanet = get_module(config, load_path)
    metanet.eval()
    if classify:
        mean, std, ci95, mean_accu, std_accu, ci95_accu = metanet.test(dataset, classify=classify)
    else:
        mean, std, ci95 = metanet.test(dataset, classify=classify)

    del metanet
    gc.collect()

    printl(model + ' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (mean, std, ci95))
    if classify:
        printl(model + ' accu on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (mean_accu, std_accu, ci95_accu))
        return mean, std, ci95, mean_accu, std_accu, ci95_accu
    else:
        return mean, std, ci95


def module_val(get_module, testset, config, classify=True):
    on_set = 'test' if config['test_or_val'] else 'val'
    test_iter = np.load(config['save_path'] + '/last_iter.npy') \
        if config['test_iter'] == -1 else config['test_iter'] * config['test_interval']
    test_iters = [test_iter + i * config['test_interval'] for i in range(-config['len_test'] + 1, 1)]

    means = []
    stds = []
    ci95s = []
    if classify:
        mean_accus = []
        std_accus = []
        ci95_accus = []
    for test_iter in test_iters:
        config['test_iter'] = test_iter
        if classify:
            mean, std, ci95, mean_accu, std_accu, ci95_accu =\
                module_test(get_module, testset, on_set, config, classify)
        else:
            mean, std, ci95 = module_test(get_module, testset, on_set, config, classify)
        printl("")
        means.append(mean)
        stds.append(std)
        ci95s.append(ci95)
        if classify:
            mean_accus.append(mean_accu)
            std_accus.append(std_accu)
            ci95_accus.append(ci95_accu)

    bestn = np.argsort(mean_accus)[::-1][:config.get('bestn', 1)] \
        if classify else np.argsort(mean_accus)[:config.get('bestn', 1)]
    models = ['_i' + str(test_iters[best]) + '.pkl' for best in bestn]
    printl(models)

    with open(os.path.join(config['save_path'], 'model_bestn.txt'), "w") as log:
        for model in models:
            log.write(model + '\n')

    printl('Iter%d is the best accub_val or loss_val: ' % bestn[0])
    printl(models[0] + ' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
        means[bestn[0]], stds[bestn[0]], ci95s[bestn[0]]))
    if classify:
        printl(models[0] + ' accu on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
            mean_accus[bestn[0]], std_accus[bestn[0]], ci95_accus[bestn[0]]))


def log_nnl(inpt, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='elementwise_mean'):
    return F.nll_loss(torch.log(inpt), target, weight, size_average, ignore_index,
                      reduce, reduction)


def nega_accu(inpt, target, reduction='elementwise_mean'):
    if reduction=='elementwise_mean':
        return - (torch.argmax(inpt, 1) == target).type(FloatTensor).mean()
    elif reduction=='none':
        return - (torch.argmax(inpt, 1) == target).type(FloatTensor)


def model_output(get_module, model, testset, config):
    '''
    if config['lossf'] == 'margin_loss' or \
       config['lossf'] == 'margin_l1_loss' or \
       config['lossf'] == 'multi_margin_loss':
        activator = lambda x:x
    elif config['lossf'] == 'cross_entropy':
        activator = nn.Softmax(dim=2)
    '''

    dataset = torch.utils.data.DataLoader(testset, batch_size=config['meta_batch_size'],
                                          shuffle=False, num_workers=8)
    load_path = os.path.join(config['save_path'], model)
    metanet = get_module(config, load_path)
    metanet.eval()

    parall_num = config.get('parall_num', config['meta_batch_size'])
    max_inter = 100000
    ys = []

    for itr, data in tqdm(enumerate(dataset)):
        feata, labela, featb, labelb = metanet.to_var(data)


        batch_size = len(labelb)
        i = 0
        while i < batch_size:
            y = metanet.net(feata[i:i+parall_num], labela[i:i+parall_num],
                            featb[i:i+parall_num]).detach()
            #ys.append(activator(y))
            ys.append(y)
            i += parall_num

        #y = metanet.net(feata, labela, featb).detach()
        #ys.append(activator(y))

        if itr >= max_inter:
            break

    del metanet
    del dataset
    gc.collect()
    return torch.cat(ys, dim=0)


def calculate_result(y_total, l_total, on_set, config, classify=True, name='', pattern='None'):
    if config['lossf'] == 'margin_loss':
        lossf = margin_loss
    elif config['lossf'] == 'margin_l1_loss':
        lossf = margin_l1_loss
    elif config['lossf'] == 'multi_margin_loss':
        lossf = multi_margin_loss
    elif config['lossf'] == 'cross_entropy':
        lossf = log_nnl if pattern != 'WE_un_unit' else F.cross_entropy

    y_size = list(y_total.size())
    y_size = [y_size[0] * y_size[1]] + y_size[2:]
    l_size = list(l_total.size())
    l_size = [l_size[0] * l_size[1]] + l_size[2:]

    lossb = lossf(y_total.view(*y_size), l_total.view(*l_size), reduction='none'
                  ).view(y_size[0], -1).mean(1)
    lossb = lossb.cpu().numpy()
    mean = np.mean(lossb, 0)
    std = np.std(lossb, 0)
    ci95 = 1.96 * std / np.sqrt(y_size[0])
    printl(name + ' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
        mean, std, ci95))

    del lossb
    gc.collect()
    if classify:
        accus = (torch.argmax(y_total, 2) == l_total).type(FloatTensor).mean(1)
        accus = accus.cpu().numpy()
        mean_accu = np.mean(accus, 0)
        std_accu = np.std(accus, 0)
        ci95_accu = 1.96 * std_accu / np.sqrt(y_size[0])
        printl(name + ' accu on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
            mean_accu, std_accu, ci95_accu))
        printl('')
        return mean, std, ci95, mean_accu, std_accu, ci95_accu
    else:
        printl('')
        return mean, std, ci95


def ensembles_test(get_module, testset, on_set, config, classify=True, load_y=False,  pattern='None'):
    if config['lossf'] == 'margin_loss' or \
                    config['lossf'] == 'margin_l1_loss' or \
                    config['lossf'] == 'multi_margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
        if pattern == 'WE_un_unit':
            activator = lambda x: x
        else:
            activator = nn.Softmax(dim=2)

    # get model name for file
    with open(os.path.join(config['save_path'], 'model_bestn.txt'), "r") as log:
        top, bottom  = log.read().strip('\n').split('--------')
        models = top.strip('\n').split('\n')

    # get label array from testset
    l_total = []
    dataset = torch.utils.data.DataLoader(testset, batch_size=config['meta_batch_size'],
                                              shuffle=False, num_workers=8)
    for itr, data in tqdm(enumerate(dataset)):
        if use_cuda:
            l_total.append(Variable(data[3]).cuda())
        else:
            l_total.append(Variable(data[3]))
    l_total = torch.cat(l_total, dim=0)

    if load_y:
        with open(os.path.join(config['save_path'], 'y_totals_test.pkl'), 'rb') as f:
            y_totals = pickle.load(f)

        y_totals = [activator(y_total) for y_total in y_totals]
        y_totals = torch.cat([y_total.unsqueeze(0) for y_total in y_totals], dim=0)

        # test performance of best model
        printl('%s is the best accub_val or loss_val: ' % models[0])
        calculate_result(y_totals[0], l_total, on_set, config, classify, models[0], pattern=pattern)
    else:
        y_totals = [model_output(get_module, models[0], testset, config)]

        print('[%d]' % 1, end='')
        # test performance of best model
        printl('%s is the best accub_val or loss_val: ' % models[0])
        calculate_result(activator(y_totals[0]), l_total, on_set, config, classify, models[0], pattern=pattern)

        for i, model in enumerate(models[1:], 2):
            print('[%d]' % i, end='')
            y_total = model_output(get_module, model, testset, config)
            y_totals.append(y_total)

        with open(os.path.join(config['save_path'], 'y_totals_test.pkl'), 'wb') as f:
            pickle.dump(y_totals, f)

        y_totals = [activator(y_total) for y_total in y_totals]
        y_totals = torch.cat([y_total.unsqueeze(0) for y_total in y_totals], dim=0)

    if pattern == 'None' or pattern == 'No_down':
        # get output of each model
        y_total = y_totals.mean(0)
        printl(str(models))
    elif pattern == 'WE' or pattern == 'WE_un_unit' \
            or pattern == 'WESA' or pattern == 'WEDE':
        with open(os.path.join(config['save_path'], 'WE.pkl'), 'rb') as f:
            weights = pickle.load(f)
        y_total = torch.sum(y_totals * weights, dim=0)
        printl(str([m + ': %0.4f' % float(w) for m, w in zip(models, weights.view(-1).cpu())]))
    elif pattern[:10] == 'No_down+WE':
        # get output of each model
        y_total1 = y_totals.mean(0)
        with open(os.path.join(config['save_path'], 'WE.pkl'), 'rb') as f:
            weights = pickle.load(f)
        y_total2 = torch.sum(y_totals * weights, dim=0)
        printl(str([m + ': %0.4f' % float(w) for m, w in zip(models, weights.view(-1).cpu())]))

    del y_totals
    gc.collect()

    # calculate ensemble result
    if pattern == 'None' or pattern == 'No_down' or \
        pattern == 'WE' or pattern == 'WE_un_unit' or pattern == 'WESA' or pattern == 'WEDE':
        printl('The result of ' + pattern + '.')
        return calculate_result(y_total, l_total, on_set, config, classify, pattern=pattern)
    elif pattern[:10] == 'No_down+WE':
        printl('The result of No_down.')
        calculate_result(y_total1, l_total, on_set, config, classify, pattern=pattern)
        printl('The result of WE.')
        return calculate_result(y_total2, l_total, on_set, config, classify, pattern=pattern)


def ensembles_val(get_module, testset, config, classify=True, load_y=False, pattern='None'):
    on_set = 'test' if config['test_or_val'] else 'val'
    test_iter = np.load(config['save_path'] + '/last_iter.npy') \
        if config['test_iter'] == -1 else config['test_iter'] * config['test_interval']
    models = ['_i' + str(test_iter + i * config['test_interval']) + '.pkl'
              for i in range(-config['len_test'] + 1, 1)]
    printl('pattern=' + pattern)

    if config['lossf'] == 'margin_loss' or \
                    config['lossf'] == 'margin_l1_loss' or \
                    config['lossf'] == 'multi_margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
        activator = nn.Softmax(dim=2)

    y_totals = []
    means = []
    stds = []
    ci95s = []
    if classify:
        mean_accus = []
        std_accus = []
        ci95_accus = []

    # get label array from testset
    l_total = []
    dataset = torch.utils.data.DataLoader(testset, batch_size=config['meta_batch_size'],
                                          shuffle=False, num_workers=8)
    for itr, data in tqdm(enumerate(dataset)):
        if use_cuda:
            l_total.append(Variable(data[3]).cuda())
        else:
            l_total.append(Variable(data[3]))
    l_total = torch.cat(l_total, dim=0)

    # get output of each model and calculate result of each model
    if load_y:
        with open(os.path.join(config['save_path'], 'y_totals.pkl'), 'rb') as f:
            if classify:
                y_totals, means, stds, ci95s, mean_accus, \
                std_accus, ci95_accus = pickle.load(f)
            else:
                y_totals, means, stds, ci95s = pickle.load(f)
    else:
        for model in models:
            y_total = model_output(get_module, model, testset, config)
            y_totals.append(y_total)

            if classify:
                mean, std, ci95, mean_accu, std_accu, ci95_accu = \
                    calculate_result(activator(y_total), l_total, on_set, config, classify, model)
                mean_accus.append(mean_accu)
                std_accus.append(std_accu)
                ci95_accus.append(ci95_accu)
            else:
                mean, std, ci95 = calculate_result(activator(y_total), l_total, on_set, config, classify)
            means.append(mean)
            stds.append(std)
            ci95s.append(ci95)

        with open(os.path.join(config['save_path'], 'y_totals.pkl'), 'wb') as f:
            if classify:
                pickle.dump((y_totals, means, stds, ci95s, mean_accus, std_accus, ci95_accus), f)
            else:
                pickle.dump((y_totals, means, stds, ci95s), f)

    # change the type of lists, then sort result by preformance
    bestn = np.argsort(mean_accus)[::-1] if classify else np.argsort(means)
    means = np.array(means)[bestn]
    stds = np.array(stds)[bestn]
    ci95s = np.array(ci95s)[bestn]
    if classify:
        mean_accus = np.array(mean_accus)[bestn]
        std_accus = np.array(std_accus)[bestn]
        ci95_accus = np.array(ci95_accus)[bestn]
    models = [models[best] for best in bestn]
    y_totals = [y_totals[best] for best in bestn]

    if classify:
        info = models, bestn, means, stds, ci95s, mean_accus, std_accus, ci95_accus
    else:
        info = models, bestn, means, stds, ci95s

    if pattern == 'None' or pattern == 'No_down' or pattern[:10] == 'No_down+WE':
        top_ensemble(config, info, on_set, y_totals, l_total, classify, pattern)
    elif pattern == 'WE' or pattern == 'WE_un_unit' or \
          pattern == 'WESA' or pattern == 'WEDE':
        # print best result of one model
        printl('Iter%d is the best accub_val or loss_val: ' % bestn[0])
        printl(models[0] + ' loss on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
            means[0], stds[0], ci95s[0]))
        if classify:
            printl(models[0] + ' accu on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
                mean_accus[0], std_accus[0], ci95_accus[0]))
        # calculate weight ensemble
        weight_ensemble(config, models, on_set, y_totals, l_total, classify, pattern)


def top_ensemble(config, info, on_set, y_totals, l_total, classify, pattern):
    if config['lossf'] == 'margin_loss' or \
                    config['lossf'] == 'margin_l1_loss' or \
                    config['lossf'] == 'multi_margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
        activator = nn.Softmax(dim=2)
    #y_totals = [activator(y_total) for y_total in y_totals]

    if classify:
        models, bestn, means, stds, ci95s, mean_accus, std_accus, ci95_accus = info
    else:
        models, bestn, means, stds, ci95s = info

    # calculate ensembles preformance
    means_e = [means[0]]
    stds_e = [stds[0]]
    ci95s_e = [ci95s[0]]
    if classify:
        mean_accus_e = [mean_accus[0]]
        std_accus_e = [std_accus[0]]
        ci95_accus_e = [ci95_accus[0]]
    models_no_down = [models[0]]
    y_current = activator(y_totals[0])
    y_totals_best = [y_totals[0]]
    num = 2
    for i, y_total in enumerate(y_totals[1:], 1):
        y_currenth = y_current + activator(y_total)
        if classify:
            mean, std, ci95, mean_accu, std_accu, ci95_accu = \
                calculate_result(y_currenth / num, l_total, on_set, config, classify,
                                 'Ensembles ' + str(num) + ' model')
            if pattern == 'No_down':
                if mean_accu < mean_accus_e[-1]:
                    continue
                else:
                    models_no_down.append(models[i])
            if pattern[:10] == 'No_down+WE':
                if mean_accu < mean_accus_e[-1]:
                    continue
                else:
                    y_totals_best.append(y_total)
                    models_no_down.append(models[i])
            mean_accus_e.append(mean_accu)
            std_accus_e.append(std_accu)
            ci95_accus_e.append(ci95_accu)
        else:
            mean, std, ci95 = calculate_result(y_currenth / num, l_total, on_set, config, classify)
            if pattern == 'No_down':
                if mean > means_e[-1]:
                    continue
                else:
                    models_no_down.append(models[i])
            if pattern[:10] == 'No_down+WE':
                if mean > means_e[-1]:
                    continue
                else:
                    y_totals_best.append(y_total)
                    models_no_down.append(models[i])
        means_e.append(mean)
        stds_e.append(std)
        ci95s_e.append(ci95)
        y_current = y_currenth
        num += 1

    # change the type of lists, and get best ensembles
    means_e = np.array(means_e)
    stds_e = np.array(stds_e)
    ci95s_e = np.array(ci95s_e)
    if classify:
        mean_accus_e = np.array(mean_accus_e)
        std_accus_e = np.array(std_accus_e)
        ci95_accus_e = np.array(ci95_accus_e)

    # print best result of one model
    printl('Iter%d is the best accub_val or loss_val: ' % bestn[0])
    printl(models[0] + ' loss on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
        means[0], stds[0], ci95s[0]))
    if classify:
        printl(models[0] + ' accu on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
            mean_accus[0], std_accus[0], ci95_accus[0]))

    if pattern == 'No_down' or pattern[:10] == 'No_down+WE':
        best_e = len(models_no_down)
        models = models_no_down
        printl('First%d is the number of models used in ensembles: ' % best_e)
    else:
        # calculate best num of ensembles
        best_e = np.argmax(mean_accus_e[::-1]) if classify else np.argmin(means_e[::-1])
        best_e = len(means_e) - 1 - best_e
        best_e += 1
        printl('Top%d is the number of models used in ensembles: ' % best_e)

    # print best result of ensembles
    printl(str(models[:best_e]))
    printl(' loss on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
        means_e[best_e - 1], stds_e[best_e - 1], ci95s_e[best_e - 1]))
    if classify:
        printl(' accu on ' + on_set + ': mean=%.6f, std=%.6f, ci95=%.6f' % (
            mean_accus_e[best_e - 1], std_accus_e[best_e - 1], ci95_accus_e[best_e - 1]))

    # calculate weight ensemble
    if pattern[:10] == 'No_down+WE':
        weight_ensemble(config, models_no_down, on_set, y_totals_best, l_total, classify, pattern)

    # save file
    with open(os.path.join(config['save_path'], 'model_bestn.txt'), "w") as log:
        for model in models[:best_e]:
            log.write(model + '\n')
        log.write('--------\n')
        for model in models[best_e:]:
            log.write(model + '\n')

    # save curve
    save_block = np.stack((means_e, stds_e, ci95s_e), axis=0)
    if classify:
        save_block = np.concatenate((save_block, np.stack(
            (mean_accus_e, std_accus_e, ci95_accus_e), axis=0)), axis=0)
    np.save(config['logdir'][:-4], save_block)

    # draw curve
    # if classify:
    #     plt_ensembles(config['logdir'][:-4], mean_accus_e, y_valu='Accuracy', picshow=config['picshow'])
    # else:
    #     plt_ensembles(config['logdir'][:-4], means_e, y_valu='Loss', picshow=config['picshow'])


def weight_ensemble(config, models, on_set, y_totals, l_total, classify, pattern):
    if config['lossf'] == 'margin_loss':
        lossf = margin_loss
    elif config['lossf'] == 'margin_l1_loss':
        lossf = margin_l1_loss
    elif config['lossf'] == 'multi_margin_loss':
        lossf = multi_margin_loss
    elif config['lossf'] == 'cross_entropy':
        if pattern == 'WE' or pattern[:10] == 'No_down+WE' or \
            pattern == 'WESA' or pattern == 'WEDE':
            lossf = log_nnl
        elif pattern == 'WE_un_unit':
            lossf = F.cross_entropy

    if config['lossf'] == 'margin_loss' or \
                    config['lossf'] == 'margin_l1_loss' or \
                    config['lossf'] == 'multi_margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
        if pattern == 'WE' or pattern[:10] == 'No_down+WE' or \
            pattern == 'WESA' or pattern == 'WEDE':
            activator = nn.Softmax(dim=2)
        elif pattern == 'WE_un_unit':
            activator = lambda x: x

    y_totals = [activator(y_total) for y_total in y_totals]

    random.seed(1)
    torch.manual_seed(1)
    if pattern=='WE' or pattern == 'No_down+WE':
        we = WeightEnsemble(len(models), lossf, unit=True)
    elif pattern == 'WE_un_unit':
        we = WeightEnsemble(len(models), lossf, unit=False)
    elif pattern == 'WESA':
        we = WESA(len(models), lossf)
    elif pattern == 'WEDE' or pattern == 'No_down+WEDE':
        we = WEDE(len(models), lossf)

    we = we.cuda() if use_cuda else we
    y_totals = torch.cat([y_total.unsqueeze(0) for y_total in y_totals], dim=0)
    weights = we(y_totals, l_total)

    # calculate ensembles preformance
    y_total = torch.sum(y_totals * weights.view(
                *([-1] + [1] * (y_totals.dim() - 1))), dim=0)
    del y_totals
    gc.collect()
    if classify:
        mean_e, std_e, ci95_e, mean_accu_e, std_accu_e, ci95_accu_e = \
            calculate_result(y_total, l_total, on_set, config, classify, 'Weight Ensembles model', pattern=pattern)
    else:
        mean_e, std_e, ci95_e = calculate_result(y_total, l_total, on_set, config, classify, pattern=pattern)

    # print result of weight ensembles
    printl(str([m + ': %0.4f' % float(w) for m, w in zip(models, weights.cpu())]))
    printl(' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
        mean_e, std_e, ci95_e))
    if classify:
        printl(' accu on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
            mean_accu_e, std_accu_e, ci95_accu_e))

    # save file and curve
    with open(os.path.join(config['save_path'], 'model_bestn.txt'), "w") as log:
        for model in models:
            log.write(model + '\n')
        log.write('--------\n')

    with open(os.path.join(config['save_path'], 'WE.pkl'), 'wb') as f:
        pickle.dump(weights, f)


class WeightEnsemble(nn.Module):
    def __init__(self, models_num, lossf, lr=1., unit=True):
        super(WeightEnsemble, self).__init__()
        self.models_num = models_num
        self.lossf = lossf
        self.weight = Parameter(torch.Tensor(models_num).type(FloatTensor))
        #self.optim = optim.SGD(self.parameters(), lr=lr, weight_decay=0., momentum=0.9, nesterov=True)
        self.optim = optim.LBFGS(self.parameters(), lr=lr)
        if unit:
            self.get_weight = self.get_unit_weight
            self.reset_unit()
        else:
            self.reset()

    def forward(self, y_totals, l_total, max_iter=5):
        y_size = list(y_totals.size())
        y_size = [y_size[0], y_size[1] * y_size[2]] + y_size[3:]
        y_totals = y_totals.view(*y_size)
        l_size = list(l_total.size())
        l_size = [l_size[0] * l_size[1]] + l_size[2:]
        l_total = l_total.view(*l_size)

        weight_size = [-1] + [1] * (y_totals.dim() - 1)

        batch_size = l_size[0]
        for itr in range(1, max_iter + 1):
            i = 0
            while i < l_size[0]:
                def get_loss():
                    global loss
                    self.optim.zero_grad()
                    weight = self.get_weight()
                    y_total = torch.sum(y_totals[:, i:i + batch_size] * weight.view(*weight_size), dim=0)
                    loss = self.lossf(y_total, l_total[i:i + batch_size])
                    loss.backward()
                    return loss
                self.optim.step(get_loss)
                #get_loss()
                #self.optim.step()
                i += batch_size

            if itr % 1 == 0:
                printl('[%d]WELoss: %.4f' % (itr, float(loss)))
        weight = self.get_weight()
        return weight.view(*([-1] + [1] * y_totals.dim())).data

    def get_weight(self):
        return self.weight

    def reset(self):
        #stdv = 1. / math.sqrt(self.models_num)
        #self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.fill_(1. / self.models_num)

    def get_unit_weight(self):
        weight = self.weight ** 2
        weight /= torch.sum(weight)
        return weight

    def reset_unit(self):
        self.weight.data.fill_(1.)


class WESA(nn.Module):
    def __init__(self, models_num, lossf, scale=0.1, alpha=0.99, T=100000., Te=1.):
        super(WESA, self).__init__()
        self.models_num = models_num
        self.lossf = nega_accu if lossf is log_nnl else lossf
        self.scale = scale
        self.alpha = alpha
        self.T = T
        self.Te = Te
        self.weight = Variable(torch.Tensor(models_num).type(FloatTensor))

    def forward(self, y_totals, l_total):
        y_size = list(y_totals.size())
        y_size = [y_size[0], y_size[1] * y_size[2]] + y_size[3:]
        y_totals = y_totals.view(*y_size)
        l_size = list(l_total.size())
        l_size = [l_size[0] * l_size[1]] + l_size[2:]
        l_total = l_total.view(*l_size)

        weight_size = [-1] + [1] * (y_totals.dim() - 1)

        T = self.T
        weight = self.weight
        self.weight_best = self.weight
        y_total = torch.sum(y_totals * weight.view(*weight_size), dim=0)
        loss = float(self.lossf(y_total, l_total))
        itr = 0
        while T > self.Te:
            weight = self.get_new_weight(weight)
            y_total = torch.sum(y_totals * weight.view(*weight_size), dim=0)
            loss_new = float(self.lossf(y_total, l_total))
            if loss_new < loss:
                self.weight_best = weight
            elif random.uniform(0., 1.) < np.exp((loss - loss_new) / T):
                pass
            else:
                weight = self.weight
                continue

            self.weight = weight
            loss = loss_new
            T *= self.alpha
            itr += 1
            if itr % 100 == 0:
                printl('[%d]WELoss: %.4f' % (itr, float(loss)))
        return self.weight_best

    def reset(self):
        self.weight.fill_(1. / self.models_num)

    def get_new_weight(self, weight):
        weight = weight + self.scale * torch.randn(self.models_num).cuda()
        weight = torch.abs(weight)
        weight /= torch.sum(weight)
        return weight


class WEDE(nn.Module):
    def __init__(self, models_num, lossf, M=None):
        super(WEDE, self).__init__()
        self.models_num = models_num
        self.M = self.models_num * 10 if M is None else M
        self.lossf = nega_accu if lossf is log_nnl else lossf
        self.pop = Variable(torch.Tensor(self.M, models_num).type(FloatTensor))
        self.reset()
        self.F_l = 0.1
        self.F_u = 0.9
        self.cr_l = 0.1
        self.cr_u = 0.6
        #######
        self.bs = 200

    def forward(self, y_totals, l_total, max_iter=100):
        y_size = list(y_totals.size())
        y_size = [1, y_size[0], y_size[1] * y_size[2]] + y_size[3:]
        y_totals = y_totals.view(*y_size)
        self.y_size = y_size[2:]
        l_size = list(l_total.size())
        l_size = [1, l_size[0] * l_size[1]] + l_size[2:]
        l_total = l_total.view(*l_size)
        self.l_size = l_size[1:]
        self.l_size_dim = len(self.l_size)
        self.whole_model()

        self.pop_size = [self.models_num] + [1] * (y_totals.dim() - 2)
        fitness = self.get_fitness(self.pop, y_totals, l_total)
        bestn = torch.argmin(fitness)
        self.weight_best = self.pop[bestn].clone()
        self.fit_best = float(fitness[bestn])
        for itr in range(1, max_iter + 1):
            U = self.mc_process(self.pop, fitness)
            fitness_U = self.get_fitness(U, y_totals, l_total)

            improve_i = fitness_U < fitness
            self.pop[improve_i] = U[improve_i]
            fitness[improve_i] = fitness_U[improve_i]

            bestc = torch.argmin(fitness)
            fit_current = float(fitness[bestc])
            if fit_current < self.fit_best:
                self.weight_best = self.pop[bestc].clone()
                self.fit_best = fit_current
            if itr % 20 == 0:
                printl('[%d]WELoss: %f' % (itr, fit_current))

        return self.get_weight(self.weight_best.unsqueeze(0)).view(*([-1] + [1] * (y_totals.dim() - 1))).data

    def batch_model(self):
        self.batch_size = self.bs
        self.parall_pop = self.M // 100
        #self.batch_size2 = self.models_num

    def whole_model(self):
        self.batch_size = self.l_size[0]
        self.parall_pop = 10
        #self.batch_size2 = 30

    def get_weight(self, pop):
        pop = pop ** 2
        pop_sum = pop.sum(dim=1, keepdim=True)
        pop_sum[pop_sum == 0.] = 1.
        pop = pop / pop_sum
        pop[(pop_sum == 0.)[:, 0]] = 1. / self.models_num
        return pop

    def get_fitness(self, pop, y_totals, l_total):
        if self.batch_size != self.l_size[0]:
            sample_i = torch.randperm(self.l_size[0])[:self.batch_size]
            y_totals = y_totals[:,:,sample_i]
            l_total = l_total[:,sample_i]

        pop_size0 = pop.size(0)
        pop = self.get_weight(pop)

        y_total = []
        i = 0
        while i < pop_size0:
            # j = 0
            # y_t = []
            # while j < self.models_num:
            #     pop_h = pop[i:i+self.batch_size1, j:j+self.batch_size2]
            #     pop_h_size = pop_h.size()
            #     y_t.append(torch.sum(y_totals[:, j:j+self.batch_size2] *
            #         pop_h.view(pop_h_size[0], pop_h_size[1], *self.pop_size[1:]) , dim=1, keepdim=True))
            #     j += self.batch_size2
            # y_total.append(torch.sum(torch.cat(y_t, dim=1),dim=1))
            pop_h = pop[i:i+self.parall_pop]
            pop_h_size = pop_h.size()
            y_total.append(torch.sum(y_totals * pop_h.view(
                pop_h_size[0], pop_h_size[1], *self.pop_size[1:]),dim=1))
            i += self.parall_pop
        y_total = torch.cat(y_total, dim=0)
        l_total = l_total.expand(pop_size0, *([-1] * self.l_size_dim)).contiguous()
        fitness = self.lossf(y_total.view(self.batch_size * pop_size0, *self.y_size[1:]),
                             l_total.view(self.batch_size * pop_size0, *self.l_size[1:]), reduction='none')
        return fitness.view(pop_size0, -1).mean(1)

    def mc_process(self, pop, fitness):
        # Mutation
        f_bmw, pop_i = self.choose_pop(fitness, 3)
        F = self.F_l + (self.F_u - self.F_l) * (f_bmw[:,1] - f_bmw[:,0]) / (f_bmw[:,2] - f_bmw[:,0])
        F = F.type(FloatTensor)
        V = pop[pop_i[:,0]] + F.unsqueeze(-1) * (pop[pop_i[:,1]] - pop[pop_i[:,2]])

        # Crossover
        F_mean = F.mean()
        F_min = F.min()
        F_max = F.max()
        cr = torch.full((self.M,), self.cr_l).type(FloatTensor)
        cr_i = F > F_mean
        cr[cr_i] += (self.cr_u - self.cr_l) * (F[cr_i] - F_min) / (F_max - F_min)
        U = pop.clone()
        uniform = torch.Tensor(self.M, self.models_num).type(FloatTensor)
        uniform.uniform_(0., 1.)
        U_i = uniform < cr.unsqueeze(-1)
        U[U_i] = V[U_i]
        return U

    def choose_pop(self, fitness, n):
        pop_i = torch.randint(1, self.M, [self.M, n]).type(torch.LongTensor)
        for i in range(n-1):
            while 1:
                indsum = 0
                for j in range(i+1):
                    ind = pop_i[:, j:j+1] == pop_i[:, i+1:]
                    pop_i[:, i+1:][ind] += 1
                    pop_i[:, i+1:][pop_i[:, i+1:] >= self.M - 1] -= self.M - 2
                    indsum += ind.sum()
                if indsum == 0:
                    break
        pop_i += torch.arange(0, self.M).unsqueeze(1)
        pop_i[pop_i >= self.M] -= self.M
        pop_i = pop_i.type(LongTensor)
        f_bmw, ind = torch.sort(fitness[pop_i], dim=1)
        return f_bmw, torch.gather(pop_i,1,ind)

    def reset(self):
        self.pop.uniform_(0., 1. / self.models_num)

'''
if __name__ == '__main__':
    M = 11
    we = WEDE(10, log_nnl, M)
    we.mc_process(we.pop, torch.randn(M))
'''

def plt_ensembles(path, vector, y_valu='Loss', picshow=True):
    len_step = (len(vector) + 1)
    x = np.arange(1, len_step)

    fig = plt.figure(num=1, figsize=(5, 4))
    plt.style.use('seaborn_grayback')

    fig.tight_layout()
    plt.xlim(0, len_step)
    plt.plot(x, vector, color='forestgreen', linestyle='-')
    #plt.scatter(x, vector, s=15, color='blue', edgecolors='black', marker='o', alpha=1.)
    plt.title('MiniImagenet')
    plt.xlabel('Ensembles Number')
    plt.ylabel(y_valu)
    plt.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path + '.png', dpi=1600)
    plt.pause(0.01)
    if picshow:
        plt.show()
    else:
        fig.clear()

'''
if __name__ == '__main__':
    path = '../../Result/mini/fuzzymeta/5way_5shot_15query/Thu-Oct-25-16-38-40-2018'
    plt_ensembles(path, np.load(path + '.npy')[3], y_valu='Accuracy')
'''

class BatchWeight(nn.Module):

    def __init__(self, num_features, num_iter, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchWeight, self).__init__()
        self.num_mv = num_iter + 1
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_means', torch.zeros(num_iter + 1, num_features))
            self.register_buffer('running_vars', torch.ones(num_iter + 1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_means.zero_()
            self.running_vars.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def reset(self, num):
        self.itr = 0 if self.track_running_stats else None

    def forward(self, input):
        if self.itr is not None:
            if self.itr < self.num_mv:
                self.running_mean = self.running_means[self.itr]
                self.running_var = self.running_vars[self.itr]
                self.itr += 1
            elif not self.training:
                self.running_mean = self.running_means[-1]
                self.running_var = self.running_vars[-1]

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(BatchWeight, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class MulLearnable(NFTModule):
    def __init__(self, alpha=1.):
        super(MulLearnable, self).__init__()
        self.alpha = Parameter(torch.Tensor([alpha]))
    def forward(self, x):
        return self.alpha * x


class MulAddLearnable(NFTModule):
    def __init__(self, alpha=1., beta=0.):
        super(MulAddLearnable, self).__init__()
        self.alpha = Parameter(torch.Tensor([alpha]))
        self.beta = Parameter(torch.Tensor([beta]))
    def forward(self, x):
        return self.alpha * x + self.beta


class Param_Enc(nn.Module):
    def __init__(self, num_features, hidden_size, channels, num_class, nkshot):
        super(Param_Enc, self).__init__()
        self.hidden_size = hidden_size
        self.channels = channels
        self.num_class = num_class
        self.nkshot = nkshot
        self.train_enc = CapsuleShare(num_features * self.nkshot, hidden_size * self.channels // 2, routings=3, retain_grad=True)
        self.test_enc = CapsuleShare(num_features, hidden_size * self.channels // 2, routings=3, retain_grad=True)
        self.avg = nn.AvgPool2d(5)
        self.bn = nn.BatchNorm1d(self.channels, affine=True)

    def forward(self, feata, labela, featb):
        featb = featb.transpose(0, 1).contiguous()
        fb_size = list(featb.size())
        featb = self.avg(featb.view(*([fb_size[0] * fb_size[1]] + fb_size[2:]))
                         ).view(fb_size[0], fb_size[1], -1).transpose(1, 2).contiguous()

        feata = feata.transpose(0, 1).contiguous()
        fa_size = list(feata.size())
        feata = self.avg(feata.view(*([fa_size[0] * fa_size[1]] + fa_size[2:]))).view(fa_size[0], fa_size[1], -1)
        fa_size = list(feata.size())
        labela = labela.transpose(0, 1).contiguous()
        labela, indices = torch.sort(labela, dim=1)
        feata = torch.gather(feata, 1, indices.view(
            *([fa_size[0], fa_size[1]] + [1] * len(fa_size[2:]))).expand_as(feata))
        feata = feata.view(fa_size[0], fa_size[1] // self.num_class, -1).transpose(1, 2).contiguous()
        hidden = torch.cat([self.train_enc(feata), self.test_enc(featb)], dim=1
                           ).view(fa_size[0], self.channels, self.hidden_size)
        return F.relu(self.bn(hidden))