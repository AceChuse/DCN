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
        if -2 <= ((self.epoch - self.milestones2) % self.interval - self.interval) <= 2:
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


class View(nn.Module):

    def __init__(self,*size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        s = x.size()
        return x.view(s[0],s[1], *self.size)

    def extra_repr(self):
        return 'size={size}'.format(**self.__dict__)


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
            ys.append(y)
            i += parall_num

        if itr >= max_inter:
            break

    del metanet
    del dataset
    gc.collect()
    return torch.cat(ys, dim=0)


def calculate_result(y_total, l_total, on_set, config, classify=True, name='', pattern='None'):
    if config['lossf'] == 'margin_loss':
        lossf = margin_loss
    elif config['lossf'] == 'cross_entropy':
        lossf = F.cross_entropy

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
    if config['lossf'] == 'margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
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

    del y_totals
    gc.collect()

    # calculate ensemble result
    if pattern == 'None' or pattern == 'No_down':
        printl('The result of ' + pattern + '.')
        return calculate_result(y_total, l_total, on_set, config, classify, pattern=pattern)


def ensembles_val(get_module, testset, config, classify=True, load_y=False, pattern='None'):
    on_set = 'test' if config['test_or_val'] else 'val'
    test_iter = np.load(config['save_path'] + '/last_iter.npy') \
        if config['test_iter'] == -1 else config['test_iter'] * config['test_interval']
    models = ['_i' + str(test_iter + i * config['test_interval']) + '.pkl'
              for i in range(-config['len_test'] + 1, 1)]
    printl('pattern=' + pattern)

    if config['lossf'] == 'margin_loss':
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

    if pattern == 'None' or pattern == 'No_down':
        top_ensemble(config, info, on_set, y_totals, l_total, classify, pattern)


def top_ensemble(config, info, on_set, y_totals, l_total, classify, pattern):
    if config['lossf'] == 'margin_loss':
        activator = lambda x: x
    elif config['lossf'] == 'cross_entropy':
        activator = nn.Softmax(dim=2)

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

    if pattern == 'No_down':
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
    if classify:
        plt_ensembles(config['logdir'][:-4], mean_accus_e, y_valu='Accuracy', picshow=config['picshow'])
    else:
        plt_ensembles(config['logdir'][:-4], means_e, y_valu='Loss', picshow=config['picshow'])


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