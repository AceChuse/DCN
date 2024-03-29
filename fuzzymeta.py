#!/usr/bin/python3.6

from torch.autograd import grad
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from collections import OrderedDict
import copy
import os

from utils import *



class MetaNet(nn.Module):
    def __init__(self, net, criterizon, optim, parall_num=1, save_path=None,
                 save_iter=10, print_iter=100):
        super(MetaNet, self).__init__()
        self.net = net.cuda() if use_cuda else net
        self.net.get_lossf(criterizon)
        self.optim = optim
        self.parall_num = parall_num

        if save_path is None:
            self.save_iter = np.inf
        else:
            self.save_path = save_path + '/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_iter = save_iter
        self.print_iter = print_iter

    def to_var(self, data):
        if use_cuda:
            return Variable(data[0]).cuda(), \
                   Variable(data[1]).cuda(), \
                   Variable(data[2]).cuda(), \
                   Variable(data[3]).cuda()
        else:
            return Variable(data[0]), Variable(data[1]), \
                   Variable(data[2]), Variable(data[3])

    def calculate(self, feata, labela, featb, labelb):
        lossa = 0.
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num]))
            loss, loss_p = self.net.inner_test(featb[i:i + self.parall_num], labelb[i:i + self.parall_num])
            loss /= batch_size
            loss_p /= batch_size
            (loss + loss_p).backward(retain_graph=False)
            lossb += float(loss)
            i += self.parall_num

        lossa /= batch_size
        return lossa, lossb

    def fit(self, trainset, valset, iters, resume_itr=0):
        loss_all = 0.
        valset = iter(valset)
        self.valset = valset
        if resume_itr == 0:
            lossa_list = []
            lossb_list = []
            lossav_list = []
            lossbv_list = []
        else:
            list_len = resume_itr // self.save_iter
            lossa_list = list(np.load(self.save_path + 'lossa_train.npy'))[:list_len]
            lossb_list = list(np.load(self.save_path + 'lossb_train.npy'))[:list_len]
            lossav_list = list(np.load(self.save_path + 'lossa_val.npy'))[:list_len]
            lossbv_list = list(np.load(self.save_path + 'lossb_val.npy'))[:list_len]

        #time_consumes = list()
        for itr, data in enumerate(trainset, resume_itr + 1):
            #starttime = time.time()
            feata, labela, featb, labelb = self.to_var(data)

            self.optim.zero_grad()
            lossa, lossb = self.calculate(feata, labela, featb, labelb)
            loss_all += lossb

            for param in self.net.parameters():
                param.grad.data.clamp_(-10, 10)

            if itr % self.save_iter == 0 and isinstance(self.optim, Scheduler):
                self.optim.see()
            else:
                self.optim.step()

            #time.sleep(5)

            # time_consumes.append(time.time() - starttime)
            # print(time_consumes[-1])
            # if len(time_consumes) == 15:
            #     np_time_consumes = np.array(time_consumes[5:])
            #     mean = np.mean(np_time_consumes)
            #     std = np.std(np_time_consumes)
            #     ci95 = 1.96 * std / np.sqrt(len(np_time_consumes))
            #     printl(''.join(['time consumes: mean=', str(mean), ', ci95=', str(ci95)]))


            if itr % self.print_iter == 0:
                printl('[%d]: %.4f' % (itr, loss_all / self.print_iter) )
                loss_all = 0.

            if itr == 1:
                lossa_list.append(lossa)
                lossb_list.append(lossb)
                lossav, lossbv = self.val(valset)
                lossav_list.append(lossav)
                lossbv_list.append(lossbv)

            elif (itr - 1) % self.save_iter == 0:
                lossa_list.append(lossa)
                lossb_list.append(lossb)
                lossav, lossbv = self.val(valset)
                lossav_list.append(lossav)
                lossbv_list.append(lossbv)

            if itr % self.save_iter == 0:
                np.save(self.save_path + 'lossa_train', np.array(lossa_list))
                np.save(self.save_path + 'lossb_train', np.array(lossb_list))
                np.save(self.save_path + 'lossa_val', np.array(lossav_list))
                np.save(self.save_path + 'lossb_val', np.array(lossbv_list))

                torch.save(self.net.state_dict(),
                       self.save_path + '_i' + str(itr) + '.pkl')
                np.save(self.save_path + 'last_iter', np.array(itr, dtype=np.int))
                if itr == self.save_iter:
                    filepath, date = os.path.split(get_resultpath())
                    with open(os.path.join(filepath, 'model_training.txt'), "w") as f:
                        f.write(date[:-4])

            if use_cuda: torch.cuda.empty_cache()

            if itr >= iters:
                break

    def val(self, valset):
        self.eval()
        feata, labela, featb, labelb = self.to_var(valset.__next__())
        lossa = 0.
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i+self.parall_num], labela[i:i+self.parall_num]))
            lossb += float(self.net.inner_test(featb[i:i+self.parall_num], labelb[i:i+self.parall_num])[0])
            i += self.parall_num

        lossa /= batch_size
        lossb /= batch_size
        self.train()
        return lossa, lossb

    def test(self, testset, classify=False, max_inter=100000):
        lossb_list = []
        accub_list = []
        num_points = 0
        for itr, data in tqdm(enumerate(testset)):
            feata, labela, featb, labelb = self.to_var(data)

            batch_size = len(labela)
            num_points += batch_size
            i = 0
            while i < batch_size:
                self.net.inner_fit(feata[i:i+self.parall_num], labela[i:i+self.parall_num])
                if classify:
                    lossbs, accubs = self.net.inner_test(featb[i:i + self.parall_num], labelb[i:i + self.parall_num],
                                                         classify=True, reduce=False)
                    lossbs = [float(lossb) for lossb in lossbs]
                    accub_list.extend([float(accub) for accub in accubs])
                else:
                    lossbs = [float(lossb) for lossb in
                              self.net.inner_test(featb[i:i+self.parall_num], labelb[i:i+self.parall_num],
                                                  reduce=False)]
                lossb_list.extend(lossbs)
                i += self.parall_num

            if itr >= max_inter:
                break
        metalosses = np.array(lossb_list)
        means = np.mean(metalosses, 0)
        stds = np.std(metalosses, 0)
        ci95 = 1.96 * stds / np.sqrt(num_points)
        if classify:
            metaaccus = np.array(accub_list)
            means_accu = np.mean(metaaccus, 0)
            stds_accu = np.std(metaaccus, 0)
            ci95_accu = 1.96 * stds_accu / np.sqrt(num_points)
            return means, stds, ci95, means_accu, stds_accu, ci95_accu
        else:
            return means, stds, ci95


class MetaGradAgree(MetaNet):
    def __init__(self, net, criterizon, optim, parall_num=1, save_path=None,
                 save_iter=10, print_iter=100):
        super(MetaGradAgree, self).__init__(net, criterizon, optim, parall_num,
                                            save_path, save_iter, print_iter)

    def calculate(self, feata, labela, featb, labelb):
        lossa = 0.
        lossb = 0.
        grads = []
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num]))
            loss, loss_p = self.net.inner_test(featb[i:i + self.parall_num], labelb[i:i + self.parall_num]
                                               , reduce=False, gagree=True)
            lossb += float(torch.sum(loss))
            loss = loss + loss_p
            for los in loss[:-1]:
                grads.append(grad(los, self.net.parameters(), retain_graph=True, create_graph=False))
            grads.append(grad(loss[-1], self.net.parameters(), retain_graph=False, create_graph=False))
            i += self.parall_num

        grads = [torch.cat([g.unsqueeze(0) for g in gs], dim=0) for gs in zip(*grads)]
        gvg = lambda x: torch.matmul(x, torch.sum(x, keepdim=True, dim=0).t())
        ws = [gvg(gs.view(batch_size, -1)) for gs in grads]
        ws = torch.cat([w.unsqueeze(0) for w in ws], dim=0).sum(dim=0)
        ws /= torch.sum(ws)
        for param, gs in zip(self.net.parameters(), grads):
            param.grad = torch.sum(gs * ws.view(*([batch_size] + [1] * (gs.dim() - 1))), dim=0)
        return lossa / batch_size, lossb / batch_size


class MetaBilevel(MetaNet):
    def __init__(self, net, criterizon, optim, parall_num=1, save_path=None,
                 save_iter=10, print_iter=100, vt=5, lam=1.):
        super(MetaBilevel, self).__init__(net, criterizon, optim, parall_num,
                                          save_path, save_iter, print_iter)
        self.vt = vt
        self.lam = lam
        if self.lam > 1. or self.lam < 0.:
            raise ValueError("Lam should in [0.,1.]!")

    def fit(self, trainset, valset, iters, resume_itr=0):
        self.iter = 0
        super().fit(trainset, valset, iters, resume_itr)

    def calculate(self, feata, labela, featb, labelb):
        lossa = 0.
        lossb = 0.
        grads = []
        batch_size = len(labela)
        i = 0
        if self.iter % self.vt == 0:
            self.calvalgrads()
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num]))
            loss, loss_p = self.net.inner_test(featb[i:i + self.parall_num], labelb[i:i + self.parall_num]
                                               , reduce=False, gagree=True)
            lossb += float(torch.sum(loss))
            loss = loss + loss_p
            for los in loss[:-1]:
                grads.append(grad(los, self.net.parameters(), retain_graph=True, create_graph=False))
            grads.append(grad(loss[-1], self.net.parameters(), retain_graph=False, create_graph=False))
            i += self.parall_num

        grads = [torch.cat([g.unsqueeze(0) for g in gs], dim=0) for gs in zip(*grads)]
        ws = [torch.matmul(gs.view(batch_size, -1), vgs)
              for gs,vgs in zip(grads, self.val_grads)]
        ws = torch.cat([w.unsqueeze(0) for w in ws], dim=0).sum(dim=0)
        #ws -= torch.min(ws)
        ws[ws < 0.] = 0.
        ws_sum = torch.sum(ws)
        ws = ws / ws_sum if ws_sum != 0. else ws
        ws = ws * self.lam + (1 - self.lam) / batch_size
        for param, gs in zip(self.net.parameters(), grads):
            param.grad = torch.sum(gs * ws.view(*([batch_size] + [1] * (gs.dim() - 1))), dim=0)

        self.iter += 1
        return lossa / batch_size, lossb / batch_size

    def calvalgrads(self):
        feata, labela, featb, labelb = self.to_var(self.valset.__next__())
        lossa = 0.
        lossb = 0.
        grads = []
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num]))
            loss, loss_p = self.net.inner_test(featb[i:i + self.parall_num], labelb[i:i + self.parall_num]
                                               , reduce=False, gagree=True)
            lossb += float(torch.sum(loss))
            loss = loss + loss_p
            for los in loss[:-1]:
                grads.append(grad(los, self.net.parameters(), retain_graph=True, create_graph=False))
            grads.append(grad(loss[-1], self.net.parameters(), retain_graph=False, create_graph=False))
            i += self.parall_num
        self.val_grads = [torch.cat([g.unsqueeze(0) for g in gs], dim=0).view(
            batch_size, -1).sum(dim=0, keepdim=True).t() for gs in zip(*grads)]


class MetaLEO(MetaNet):
    def __init__(self, net, criterizon, optim, parall_num=1, save_path=None,
                 save_iter=10, print_iter=100):
        super(MetaLEO, self).__init__(net, criterizon, optim, parall_num,
                                      save_path, save_iter, print_iter)

    def calculate(self, feata, labela, featb, labelb):
        lossa = 0.
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i + self.parall_num],
                                              labela[i:i + self.parall_num],
                                              featb[i:i + self.parall_num]))
            loss, loss_p = self.net.inner_test(labelb[i:i + self.parall_num])
            loss /= batch_size
            loss_p /= batch_size
            (loss + loss_p).backward(retain_graph=False)
            lossb += float(loss)
            i += self.parall_num

        lossa /= batch_size
        return lossa, lossb

    def val(self, valset):
        self.eval()
        feata, labela, featb, labelb = self.to_var(valset.__next__())
        lossa = 0.
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            lossa += float(self.net.inner_fit(feata[i:i+self.parall_num],
                                              labela[i:i+self.parall_num],
                                              featb[i:i + self.parall_num]))
            lossb += float(self.net.inner_test(labelb[i:i+self.parall_num])[0])
            i += self.parall_num

        lossa /= batch_size
        lossb /= batch_size
        self.train()
        return lossa, lossb

    def test(self, testset, classify=False, max_inter=100000):
        lossb_list = []
        accub_list = []
        num_points = 0
        for itr, data in tqdm(enumerate(testset)):
            feata, labela, featb, labelb = self.to_var(data)

            batch_size = len(labela)
            num_points += batch_size
            i = 0
            while i < batch_size:
                self.net.inner_fit(feata[i:i+self.parall_num],
                                   labela[i:i+self.parall_num],
                                   featb[i:i + self.parall_num])
                if classify:
                    lossbs, accubs = self.net.inner_test(labelb[i:i + self.parall_num],
                                                         classify=True, reduce=False)
                    lossbs = [float(lossb) for lossb in lossbs]
                    accub_list.extend([float(accub) for accub in accubs])
                else:
                    lossbs = [float(lossb) for lossb in
                              self.net.inner_test(labelb[i:i+self.parall_num], reduce=False)]
                lossb_list.extend(lossbs)
                i += self.parall_num

            if itr >= max_inter:
                break
        metalosses = np.array(lossb_list)
        means = np.mean(metalosses, 0)
        stds = np.std(metalosses, 0)
        ci95 = 1.96 * stds / np.sqrt(num_points)
        if classify:
            metaaccus = np.array(accub_list)
            means_accu = np.mean(metaaccus, 0)
            stds_accu = np.std(metaaccus, 0)
            ci95_accu = 1.96 * stds_accu / np.sqrt(num_points)
            return means, stds, ci95, means_accu, stds_accu, ci95_accu
        else:
            return means, stds, ci95


class FuzzyNet(nn.Module):
    def __init__(self, param_size, weight_size, Rules,
                 nums, set_num, inner_num, inner_lr, match, layers):
        super(FuzzyNet, self).__init__()
        self.net = nn.Sequential(*layers)
        self.net_C = (copy.deepcopy(self.net),)
        self.len_rules = sum(nums)
        self.Rules = Rules
        self.nums = nums

        '''
        self.match = nn.Sequential(nn.Conv2d(1, set_num, kernel_size=3),
                                   nn.ReLU(),
                                   nn.Conv2d(set_num, set_num, kernel_size=3, stride=2, padding=1),
                                   nn.AdaptiveAvgPool2d((1, 1)))
        '''

        self.match = match

        self.inner_num = inner_num
        self.inner_lr = inner_lr

        self.len_fl = 0
        self.weightn_list = []
        self.fuzzylayers = OrderedDict()
        for i, module in enumerate(self.net._modules.values()):
            if isinstance(module,Fuzzy2D):
                self.fuzzylayers[str(i)] = module
                module.get_setnum(self.len_rules, self.match)
                self.len_fl += module.in_channals * module.out_channals
                self.weightn_list.append(module.in_channals * module.out_channals)

        self.wnames = []
        self.mdict = []
        self.fuzzylayers_C = OrderedDict()
        for i, module in enumerate(self.net_C[0]._modules.values()):
            if isinstance(module,Fuzzy2D):
                self.fuzzylayers_C[str(i)] = module
                module.get_setnum(self.len_rules, self.match)
            for name, param in module.named_parameters():
                self.mdict.append(module.__dict__)
                self.wnames.append(name)
            module._parameters = OrderedDict()
        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * (self.len_w + 1)
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)

        if 2**set_num != self.len_rules:
            raise Exception('The number of rules is not equal to the number of divide regions.')

        param_size = (param_size[1], param_size[0])
        self.param_size = param_size
        weight_size = (weight_size[1], weight_size[0])
        self.weight_size = weight_size

        self.rules = Rule_list(self.Rules, self.nums, param_size, weight_size)
        self.param = Parameter(torch.Tensor(self.len_fl, 1, param_size[0], param_size[1]))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(sum(self.param.size())/self.len_fl)
        self.param.data.uniform_(-stdv, stdv)

    def init_inner(self):
        i = 0
        for m, m_C in zip(self.net._modules.values(), self.net_C[0]._modules.values()):
            for name, p in m.named_parameters():
                self.use_wlist[i] = p * Variable(torch.ones(1)).type(FloatTensor)
                m_C.__setattr__(name, self.use_wlist[i])
                i += 1

        self.param_inner = self.param * Variable(torch.ones(1)).type(FloatTensor)
        self.use_wlist[-1] = self.param_inner

    def create_weights(self):
        self.weights = self.rules(self.param_inner)

        # self.weights = F.upsample(self.param_inner, self.weight_size, mode='bilinear', align_corners=True)
        index = 0
        for weightn, layer in zip(self.weightn_list, self.fuzzylayers_C.values()):
            layer.get_weights(self.weights[index:index + weightn])
            index += weightn

    def forward(self, input):
        for module in self.net._modules.values():
            input = module(input)
        return input

    def get_lossf(self, lossf):
        self.lossf = lossf

    def inner_fit(self, feat, label):
        self.init_inner()
        for _ in range(self.inner_num):
            self.create_weights()
            loss = self.lossf(self.net_C[0](feat), label)
            grads = grad(loss,self.use_wlist,create_graph=True)
            for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
                m[wn] = m[wn] - self.inner_lr * g
                self.use_wlist[i] = m[wn]
            self.param_inner = self.param_inner - self.inner_lr * grads[-1]
            self.use_wlist[-1] = self.param_inner
        return loss

    def inner_test(self, feat, label):
        self.create_weights()
        #self.net_C[0].eval()
        return self.lossf(self.net_C[0](feat), label)

'''
def main():
    net = FuzzyNet(param_size=(3,3),
                   weight_size = (28 * 28, 28 * 28),
                   Rules=(RInception,),
                   nums=(16,),
                   set_num=4,
                   inner_num=5,
                   inner_lr=0.01,
                   layers=(Fuzzy2D((28,28),(28,28),3,3),))
    print([x for x in net])
'''

def to_parall(m):
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
    if type(m) == NochoseBase:
        return NochoseParall()
    if type(m) == NCh2D:
        return NCh2DParall()
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
    if type(m) == FuzzyBlock2D or type(m) == CNNFuzzyBlock2D:
        return FuzzyBlock2DParall(m)
    if type(m) == Fuzzy2D333:
        return Fuzzy2D333Parall(m)
    if type(m) == FuzzyDense2D:
        return FuzzyDense2DParall(m)

    if type(m) == FFCNN2D:
        return FFCNN2DParall()
    elif type(m) == FCh2DSP:
        return FCh2DSPParall()
    elif type(m) == FCh2D1M:
        return FCh2D1MParall()
    elif type(m) == FCh2DAll:
        return FCh2DAllParall()
    elif type(m) == FCh2DMH:
        return FCh2DMHParall()
    elif type(m) == FCh2D:
        return FCh2DParall()
    elif type(m) == F2DEach:
        return F2DEachParall()
    elif type(m) == FCNN2D:
        return FCNN2DParall()
    elif type(m) == Fuzzy2D:
        return Fuzzy2DParall()

    return sameparall(m)


class _MetaBase(nn.Module):
    def __init__(self):
        super(_MetaBase, self).__init__()
        self.nonfinetune = None
        self.lam = None
        self.fuzzylayers_C = OrderedDict()

    def get_lossf(self, lossf):
        self.lossf = lossf

    def init_inner(self, num):
        i = 0
        for m, m_C in zip(self.net._modules.values(), self.net_C[0]._modules.values()):
            m_C.get_parameters(m, num)
            if isinstance(m, FuzzyBlock):
                for m_s,m_C_s in zip(m._modules.values(), m_C._modules.values()):
                    if (isinstance(m_s, FCNN2D) or isinstance(m_s, NCh2D)) and hasattr(m_s, 'bw'):
                        continue
                    for name, _ in m_s.named_parameters():
                        self.use_wlist[i] = getattr(m_C_s, name)
                        i += 1
                continue
            if (isinstance(m, FCNN2D) or isinstance(m, NCh2D)) and hasattr(m, 'bw'):
                continue
            if isinstance(m, NFTModule):
                continue
            for name, _ in m.named_parameters():
                self.use_wlist[i] = getattr(m_C, name)
                i += 1

        self.net_C[0].inner_train()

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
        self.loss_p, self.feat = self.abstract1(feat)
        feat = self.feat

        self.init_inner(self.num)
        for i in range(self.inner_num):
            self.create_weights(self.num)
            output = self.net_C[0](feat)
            if i == 0:
                output_size = list(output.size())
                output_size = [output_size[0] * output_size[1]] + output_size[2:]
            loss = self.num * self.lossf(output.view(*output_size), label)

            grads = grad(loss, self.use_wlist, create_graph=self.training)
            self.inner_update(grads)
        return loss

    def inner_update(self, grads):
        pass

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(feat)
            if self.lam is not None:
                code_s = torch.sigmoid(feat)
                sp = self.rho * torch.log(self.rho / (code_s + 1e-16)) + (1. - self.rho) * torch.log(
                    (1. - self.rho) / (1. - code_s + 1e-16))

                loss_p = self.lam * torch.sum(sp)
                feat = F.relu(feat)
            else:
                loss_p = 0.
            feat = feat if self.training else feat.detach()
            return loss_p, feat
        else:
            return 0., feat

    def abstract1(self, feat):
        self.abstract1 = self.abstract
        return self.abstract(feat)

    def abstract2(self, feat):
        self.loss_p_size = self.nkshot + self.nkquery
        return self.abstract(feat)

    def inner_test(self, feat, label, classify=False, reduce=True, gagree=False):
        self.feat_size = feat.size()
        self.num = self.feat_size[0]
        self.nkquery = self.feat_size[1]
        feat = feat.transpose(0, 1).contiguous()
        self.label_size = list(label.size())
        self.label_size = [self.label_size[0] * self.label_size[1]] + self.label_size[2:]
        self.label = label.transpose(0, 1).contiguous()
        label = self.label.view(*self.label_size)
        loss_p, feat = self.abstract2(feat)
        loss_p += self.loss_p
        self.loss_p = 0.
        self.create_weights(self.num)
        self.net_C[0].inner_eval()

        output = self.net_C[0](feat)
        output_size = list(output.size())
        output_size = [output_size[0] * output_size[1]] + output_size[2:]
        if reduce:
            return self.num * self.lossf(output.view(*output_size), label),\
                   loss_p / self.loss_p_size
        else:
            loss = self.lossf(output.view(*output_size), label, reduction='none')
            loss = loss.view(-1, self.num)
            if gagree:
                return loss.mean(0), loss_p
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


class MAML(_MetaBase):
    def __init__(self, inner_num, inner_lr, layers):
        super(MAML, self).__init__()
        if isinstance(layers[0], NonFineTune):
            self.nonfinetune = layers[0]
            self.net = nn.Sequential(*layers[1:])
        else:
            self.net = nn.Sequential(*layers)
        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.net_C = [SequentialParall()]
        self.wnames = []
        self.mdict = []

        for i, module in enumerate(self.net._modules.values()):
            moduleparall = to_parall(module)
            self.net_C[0].add_module(str(i), moduleparall)

            for name, param in module.named_parameters():
                self.mdict.append(moduleparall)
                self.wnames.append(name)

        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * self.len_w
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads, self.mdict, self.wnames):
            h = m.__dict__[wn] - self.inner_lr * g
            setattr(m, wn, h)
            self.use_wlist[i] = h

    def extra_repr(self):
        s = ('inner_lr={inner_lr}, inner_num={inner_num}, ')
        return s.format(**self.__dict__)


class NochoseNetParall(_MetaBase):
    def __init__(self, param_size, weight_size, Rule, inner_num,
                 inner_lr, inner_wd, layers, lam=None, rho=0.2, init_param=True):
        super(NochoseNetParall, self).__init__()
        if isinstance(layers[0], NonFineTune):
            self.nonfinetune = layers[0]
            self.net = nn.Sequential(*layers[1:])
        else:
            self.net = nn.Sequential(*layers)
        self.lam = lam
        self.rho = rho
        len_rules = 1
        self.len_rules = len_rules
        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.net_C = [SequentialParall()]
        self.len_fl = 0
        self.weightn_list = []
        self.fuzzylayers = OrderedDict()
        self.wnames = []
        self.mdict = []
        self.bws = []

        for i, module in enumerate(self.net._modules.values()):
            if isinstance(module, NochoseBase):
                self.fuzzylayers[str(i)] = module
                module.get_setnum(self.len_rules)
                self.len_fl += module.num_weight
                self.weightn_list.append(module.num_weight)

            if isinstance(module, FuzzyBlock):
                j = 0
                for m in module._modules.values():
                    if isinstance(m, NochoseBase):
                        self.fuzzylayers[str(i)+str(j)] = m
                        m.get_setnum(self.len_rules)
                        self.len_fl += m.num_weight
                        self.weightn_list.append(m.num_weight)
                    j += 1

            moduleparall = to_parall(module)
            self.net_C[0].add_module(str(i), moduleparall)

            if isinstance(module, NochoseBase):
                self.fuzzylayers_C[str(i)] = moduleparall

            if isinstance(module, FuzzyBlock):
                j = 0
                for m, mP in zip(module._modules.values(),moduleparall._modules.values()):
                    if isinstance(m, NochoseBase):
                        self.fuzzylayers_C[str(i) + str(j)] = mP
                    j += 1

                    if isinstance(m, MetaDropout):
                        self.bws.append(m)

                    if isinstance(m, NCh2D) and hasattr(m, 'bw'):
                        if isinstance(m.bw, BatchWeight):
                            self.bws.append(m.bw)
                        continue

                    for name, param in m.named_parameters():
                        self.mdict.append(mP.__dict__)
                        self.wnames.append(name)

                continue

            if isinstance(module, MetaDropout):
                self.bws.append(module)

            if isinstance(module, NCh2D) and hasattr(module, 'bw'):
                if isinstance(module.bw, BatchWeight):
                    self.bws.append(module.bw)
                continue

            if isinstance(module, NFTModule):
                continue

            for name, param in module.named_parameters():
                self.mdict.append(moduleparall.__dict__)
                self.wnames.append(name)

        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * (self.len_w + 1)
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)

        self.param_size = param_size
        if isinstance(weight_size, int):
            self.weight_size = weight_size
        else:
            weight_size = (weight_size[1], weight_size[0])
            self.weight_size = weight_size

        self.rules = Rule(len_rules, param_size, weight_size, inner_num, self.len_fl)

        for i, module in enumerate(self.rules._modules.values()):
            if isinstance(module, BatchWeight) or isinstance(module, MetaDropout):
                self.bws.append(module)
        self.bws = tuple(self.bws)

        if init_param:
            self.param = Parameter(torch.Tensor(self.len_fl, param_size))
            self.reset_parameters()

        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)
        self.lr_pfl = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pfl.data.fill_(self.inner_lr)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)
        self.lr_pflwed = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pflwed.data.fill_(self.inner_lr * self.inner_wd)

    def reset_parameters(self):
        stdv = 1. / np.sqrt(sum(self.param.size()) / self.len_fl)
        self.param.data.uniform_(-stdv, stdv)

    def init_inner(self, num):
        super().init_inner(num)
        self.rules.inner_train()
        for bw in self.bws:
            bw.reset(num)
        self.init_param_inner(num)

        self.lr_pfl_inner = self.lr_pfl.clone().repeat(num, 1)
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(num, 1)

    def init_param_inner(self, num):
        self.param_inner = self.param.clone()
        self.param_inner = self.param_inner.repeat(num, 1)

    def create_weights(self, num):
        self.use_wlist[-1] = self.param_inner

        self.weights = self.rules(self.param_inner, num).view(
            num, self.len_fl, self.len_rules, self.weight_size[0], self.weight_size[1])

        index = 0
        for weightn, layer in zip(self.weightn_list, self.fuzzylayers_C.values()):
            layer.get_weights(self.weights[:, index:index + weightn])
            index += weightn

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)

    def inner_test(self, feat, label, classify=False, reduce=True, gagree=False):
        self.rules.inner_eval()
        return super().inner_test(feat, label, classify, reduce, gagree)

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
        self.rules.inner_eval()
        return self.net_C[0](featb).transpose(0, 1).contiguous()


class FNetParall(_MetaBase):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, match, layers, lam=None, rho=0.2, init_param=True):
        super(FNetParall, self).__init__()
        if isinstance(layers[0], NonFineTune):
            self.nonfinetune = layers[0]
            self.net = nn.Sequential(*layers[1:])
        else:
            self.net = nn.Sequential(*layers)
        self.lam = lam
        self.rho = rho
        self.len_rules = len_rules
        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.match = match
        self.net_C = [SequentialParall()]
        self.len_fl = 0
        self.weightn_list = []
        self.fuzzylayers = OrderedDict()
        self.wnames = []
        self.mdict = []
        self.bws = []

        for i, module in enumerate(self.net._modules.values()):
            if isinstance(module, FuzzyBase):
                self.fuzzylayers[str(i)] = module
                module.get_setnum(self.len_rules, self.match)
                self.len_fl += module.num_weight
                self.weightn_list.append(module.num_weight)

            if isinstance(module, FuzzyBlock):
                j = 0
                for m in module._modules.values():
                    if isinstance(m, FuzzyBase):
                        self.fuzzylayers[str(i)+str(j)] = m
                        m.get_setnum(self.len_rules, self.match)
                        self.len_fl += m.num_weight
                        self.weightn_list.append(m.num_weight)
                    j += 1

            moduleparall = to_parall(module)
            self.net_C[0].add_module(str(i), moduleparall)

            if isinstance(module, FuzzyBase):
                self.fuzzylayers_C[str(i)] = moduleparall

            if isinstance(module, FuzzyBlock):
                j = 0
                for m, mP in zip(module._modules.values(),moduleparall._modules.values()):
                    if isinstance(m, FuzzyBase):
                        self.fuzzylayers_C[str(i) + str(j)] = mP
                    j += 1

                    if isinstance(m, MetaDropout):
                        self.bws.append(m)

                    if isinstance(m, FCNN2D) and hasattr(m, 'bw'):
                        if isinstance(m.bw, BatchWeight):
                            self.bws.append(m.bw)
                        continue

                    for name, param in m.named_parameters():
                        self.mdict.append(mP.__dict__)
                        self.wnames.append(name)

                continue

            if isinstance(module, MetaDropout):
                self.bws.append(module)

            if isinstance(module, FCNN2D) and hasattr(module, 'bw'):
                if isinstance(module.bw, BatchWeight):
                    self.bws.append(module.bw)
                continue

            if isinstance(module, NFTModule):
                continue

            for name, param in module.named_parameters():
                self.mdict.append(moduleparall.__dict__)
                self.wnames.append(name)

        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * (self.len_w + 1)
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)

        if 2 ** set_num != self.len_rules:
            raise Exception('The number of rules is not equal to the number of divide regions.')

        self.param_size = param_size
        if isinstance(weight_size, int):
            self.weight_size = weight_size
        else:
            weight_size = (weight_size[1], weight_size[0])
            self.weight_size = weight_size

        self.rules = Rule(len_rules, param_size, weight_size, inner_num, self.len_fl)

        for i, module in enumerate(self.rules._modules.values()):
            if isinstance(module, BatchWeight) or isinstance(module, MetaDropout):
                self.bws.append(module)
        self.bws = tuple(self.bws)

        if init_param:
            self.param = Parameter(torch.Tensor(self.len_fl, param_size))
            self.reset_parameters()
        if isinstance(self.match, EachMatch):
            self.match.init_params()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(sum(self.param.size()) / self.len_fl)
        self.param.data.uniform_(-stdv, stdv)

    def init_inner(self, num):
        super().init_inner(num)
        self.rules.inner_train()
        for bw in self.bws:
            bw.reset(num)
        self.init_param_inner(num)

    def init_param_inner(self, num):
        self.param_inner = self.param.clone()
        self.param_inner = self.param_inner.repeat(num, 1)

    def create_weights(self, num):
        self.use_wlist[-1] = self.param_inner

        self.weights = self.rules(self.param_inner, num).view(
            num, self.len_fl, self.len_rules, self.weight_size[0], self.weight_size[1])

        index = 0
        for weightn, layer in zip(self.weightn_list, self.fuzzylayers_C.values()):
            layer.get_weights(self.weights[:, index:index + weightn])
            index += weightn

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - self.inner_lr * g
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - self.inner_lr * grads[-1]

    def inner_test(self, feat, label, classify=False, reduce=True, gagree=False):
        self.rules.inner_eval()
        return super().inner_test(feat, label, classify, reduce, gagree)

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
        self.rules.inner_eval()
        return self.net_C[0](featb).transpose(0, 1).contiguous()

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, '
             'param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class FLeRParall(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, match, layers, lam=None, rho=0.2, init_param=True):
        super(FLeRParall, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho, init_param)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)
        self.lr_pfl = Parameter(torch.zeros(self.len_fl,1))
        self.lr_pfl.data.fill_(self.inner_lr)

    def init_inner(self, num):
        super().init_inner(num)
        self.lr_pfl_inner = self.lr_pfl.clone().repeat(num, 1)

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - self.lr_pw[i] * g
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - self.lr_pfl_inner * grads[-1]


class FWeDParall(FLeRParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam=None, rho=0.2, init_param=True):
        super(FWeDParall, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho, init_param)
        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)
        self.lr_pflwed = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pflwed.data.fill_(self.inner_lr * self.inner_wd)

    def init_inner(self, num):
        super().init_inner(num)
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(num, 1)

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


def clip_grad_norm(grads, max_norm, norm_type=2):
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    grads = list(filter(lambda g: g is not None, grads))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.abs().max() for g in grads)
    else:
        total_norm = 0
        for g in grads:
            g_norm = g.norm(norm_type)
            total_norm += g_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    return total_norm


class FWeDCGN(FWeDParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam=None, rho=0.2, init_param=True):
        super(FWeDCGN, self).__init__(param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam, rho, init_param)

    def inner_update(self, grads):
        clip_grad_norm(grads, max_norm=1.)

        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)


class FWeDCGNM(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam=None, rho=0.2, init_param=True):
        super(FWeDCGNM, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho, init_param)

        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(inner_num, self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)
        self.lr_pfl = Parameter(torch.zeros(inner_num, self.len_fl, 1))
        self.lr_pfl.data.fill_(self.inner_lr)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(inner_num, self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)
        self.lr_pflwed = Parameter(torch.zeros(inner_num, self.len_fl, 1))
        self.lr_pflwed.data.fill_(self.inner_lr * self.inner_wd)

    def init_inner(self, num):
        super().init_inner(num)
        self.lr_pfl_inner = self.lr_pfl.clone().repeat(1, num, 1)
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(1, num, 1)
        self.up_i = 0

    def inner_update(self, grads):
        clip_grad_norm(grads, max_norm=1.)

        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[self.up_i, i] * g + self.lr_pwed[self.up_i, i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner[self.up_i] * grads[-1] + self.lr_pflwed_inner[self.up_i] * self.param_inner)
        self.up_i += 1

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class RLWeD_Enc(nn.Module):
    def __init__(self, num_features, hidden_size, num_class):
        super(RLWeD_Enc, self).__init__()
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.train_enc = nn.Linear(num_features, hidden_size, bias=False)
        self.test_enc = nn.Linear(num_features, hidden_size, bias=False)
        self.avg = nn.AvgPool2d(5)
        self.bn_tr = nn.BatchNorm1d(hidden_size * num_class, affine=True)
        self.bn_test = nn.BatchNorm1d(hidden_size, affine=True)

    def forward(self, feata, labela, featb):
        fb_size = list(featb.size())
        featb = self.avg(featb.view(*([fb_size[0] * fb_size[1]] + fb_size[2:])))
        featb = self.test_enc(featb.view(fb_size[0] * fb_size[1], -1)
                              ).view(fb_size[0], fb_size[1], -1)

        fa_size = list(feata.size())
        feata = self.avg(feata.view(*([fa_size[0] * fa_size[1]] + fa_size[2:])))
        feata = self.train_enc(feata.view(fa_size[0] * fa_size[1], -1)
                               ).view(fa_size[0], fa_size[1], -1)

        fa_size = list(feata.size())
        labela, indices = torch.sort(labela, dim=0)
        feata = torch.gather(feata, 0, indices.view(
            *([fa_size[0], fa_size[1]] + [1] * len(fa_size[2:]))).expand_as(feata))
        feata = self.bn_tr(feata.view(self.num_class, fa_size[0] // self.num_class, fa_size[1], -1
                                      ).mean(dim=1).transpose(0, 1).reshape(fa_size[1], -1))

        featb = self.bn_test(featb.mean(dim=0))

        return F.relu(torch.cat([feata, featb], dim=1))


class FEncWeD(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, rlwed_enc, lam=None, rho=0.2, init_param=True):
        super(FEncWeD, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho, init_param)
        self.rlwed_enc = rlwed_enc
        self.r_enc_last = nn.Linear(self.rlwed_enc.hidden_size * (self.rlwed_enc.num_class + 1),
                                    (self.len_w + self.len_fl) * 2)
        self.r_enc_last.weight.data.zero_()
        self.r_enc_last.bias.data.zero_()

        self.inner_lr = inner_lr
        self.inner_wd = inner_wd

    def init_inner(self, num):
        lr_wed = self.r_enc_last(self.rlwed_enc(self.feat, self.label, self.featb)
                                 ).view(-1, 2, self.len_w + self.len_fl)
        lr_pw = lr_wed[:, 0, :self.len_w]
        lr_pwed = lr_wed[:, 1, :self.len_w]

        self.lr_pfl_inner = lr_wed[:, 0, self.len_w:].reshape(-1, 1) + self.inner_lr
        self.lr_pflwed_inner = lr_wed[:, 1, self.len_w:].reshape(-1, 1) + self.inner_lr * self.inner_wd

        self.lr_pw = [[]] * self.len_w
        self.lr_pwed = [[]] * self.len_w

        i = 0
        for m, m_C in zip(self.net._modules.values(), self.net_C[0]._modules.values()):
            m_C.get_parameters(m, num)
            if isinstance(m, FuzzyBlock):
                for m_s,m_C_s in zip(m._modules.values(), m_C._modules.values()):
                    if isinstance(m_s, FCNN2D) and hasattr(m_s, 'bw'):
                        continue
                    for name, _ in m_s.named_parameters():
                        self.use_wlist[i] = getattr(m_C_s, name)
                        self.lr_pw[i] = m_C_s.repeat_lr(lr_pw[:,i], name) + self.inner_lr
                        self.lr_pwed[i] = m_C_s.repeat_lr(lr_pwed[:,i], name) + self.inner_lr * self.inner_wd
                        i += 1
                continue
            if isinstance(m, FCNN2D) and hasattr(m, 'bw'):
                continue
            if isinstance(m, NFTModule):
                continue
            for name, _ in m.named_parameters():
                self.use_wlist[i] = getattr(m_C, name)
                self.lr_pw[i] = m_C.repeat_lr(lr_pw[:,i], name) + self.inner_lr
                self.lr_pwed[i] = m_C.repeat_lr(lr_pwed[:,i], name) + self.inner_lr * self.inner_wd
                i += 1

        self.net_C[0].inner_train()
        self.rules.inner_train()
        for bw in self.bws:
            bw.reset(num)
        self.init_param_inner(num)

    def init_param_inner(self, num):
        self.param_inner = self.param.clone()
        self.param_inner = self.param_inner.repeat(num, 1)

    def inner_update(self, grads):
        clip_grad_norm(grads, max_norm=1.)

        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(torch.cat([feat, self.featb], dim=0))
            feat = feat if self.training else feat.detach()
            self.featb = feat[self.nkshot:]
            return feat[:self.nkshot]
        else:
            return feat

    def abstract1(self, feat):
        return 0., self.abstract(feat)

    def inner_fit(self, feata, label, featb):
        self.featb = featb.transpose(0, 1).contiguous()
        return super().inner_fit(feata, label)

    def inner_test(self, label, classify=False, reduce=True, gagree=False):
        self.rules.inner_eval()
        feat = self.featb
        self.label_size = list(label.size())
        self.label_size = [self.label_size[0] * self.label_size[1]] + self.label_size[2:]
        self.label = label.transpose(0, 1).contiguous()
        label = self.label.view(*self.label_size)
        self.loss_p = 0.
        self.create_weights(self.num)
        self.net_C[0].inner_eval()

        output = self.net_C[0](feat)
        output_size = list(output.size())
        output_size = [output_size[0] * output_size[1]] + output_size[2:]
        if reduce:
            return self.num * self.lossf(output.view(*output_size), label), 0.
        else:
            loss = self.lossf(output.view(*output_size), label, reduction='none')
            loss = loss.view(-1, self.num)
            if gagree:
                return loss.mean(0), 0.
            elif classify:
                self.label = self.label.view(self.label_size[0] // self.num, self.num)
                c = (torch.argmax(output, 2) == self.label).type(FloatTensor)
                return loss.mean(0), c.mean(0)
            else:
                return loss.mean(0)

    def forward(self, feata, labela, featb):
        self.inner_fit(feata, labela, featb)
        feat_size = featb.size()
        self.num = feat_size[0]
        self.nkquery = feat_size[1]
        self.create_weights(self.num)
        self.net_C[0].inner_eval()
        self.rules.inner_eval()
        return self.net_C[0](self.featb).transpose(0, 1).contiguous()


class FWeDReptile(FWeDParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam=None, rho=0.2, init_param=True):
        super(FWeDReptile, self).__init__(param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, lam, rho, init_param)

    def abstract1(self, feat):
        if self.nonfinetune is not None:
            return 0., self.nonfinetune(feat)
        else:
            return 0., feat

    def init_inner(self, num):
        super().init_inner(num)
        self.use_wlist_ = [w for w in self.use_wlist[:-1]]
        self.param_inner_ = self.param_inner

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)

    def abstract2(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(feat)
        loss_p = 0.
        for w_, w in zip(self.use_wlist_, self.use_wlist[:-1]):
            loss_p = loss_p + torch.sum((w_ - w) ** 2)
        loss_p = loss_p + torch.sum((self.param_inner_ - self.param_inner) ** 2)
        self.loss_p_size = self.num
        return self.lam * loss_p, feat


class FWeDSigmal(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(FWeDSigmal, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(np.sqrt(self.inner_lr))
        self.lr_pfl = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pfl.data.fill_(np.sqrt(self.inner_lr))

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(-torch.log(torch.Tensor(1. / (self.inner_lr * self.inner_wd) - 1.))[0])
        self.lr_pflwed = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pflwed.data.fill_(-torch.log(torch.Tensor(1. / (self.inner_lr * self.inner_wd) - 1.))[0])

    def init_inner(self, num):
        super().init_inner(num)
        self.lr_pfl_inner = self.lr_pfl.clone().repeat(num, 1)
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(num, 1)

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (torch.sigmoid(self.lr_pw[i]) * g + torch.sigmoid(self.lr_pwed[i]) * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            torch.sigmoid(self.lr_pfl_inner) * grads[-1] + torch.sigmoid(self.lr_pflwed_inner) * self.param_inner)

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class FWeDEParall(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(FWeDEParall, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.inner_num, self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)
        self.lr_pfl = Parameter(torch.zeros(self.inner_num, self.len_fl, 1))
        self.lr_pfl.data.fill_(self.inner_lr)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.inner_num, self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)
        self.lr_pflwed = Parameter(torch.zeros(self.inner_num, self.len_fl, 1))
        self.lr_pflwed.data.fill_(self.inner_lr * self.inner_wd)

    def init_inner(self, num):
        super().init_inner(num)
        self.lr_pfl_inner = self.lr_pfl.clone().repeat(1, num, 1)
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(1, num, 1)
        self.step = 0

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.lr_pw[self.step][i] * g + self.lr_pwed[self.step][i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            self.lr_pfl_inner[self.step] * grads[-1] + self.lr_pflwed_inner[self.step] * self.param_inner)
        self.step += 1

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class FWeDMoreHalf(FWeDParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(FWeDMoreHalf, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam, rho)
        self.create_weights_save = self.create_weights

    def empty(self, num):
        pass

    def inner_fit(self, feat, label):
        loss = super().inner_fit(feat, label)
        self.create_weights(self.num)
        self.net_C[0](self.feat)
        self.create_weights = self.empty
        return loss

    def inner_test(self, feat, label, classify=False, reduce=True, gagree=False):
        result = super().inner_test(feat, label, classify, reduce, gagree)
        self.create_weights = self.create_weights_save
        return result


class FWeDSparseM(FWeDParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(FWeDSparseM, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam, rho)

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(feat)
            feat = feat if self.training else feat.detach()
            return feat
        else:
            return feat

    def abstract1(self, feat):
        return 0., self.abstract(feat)

    def abstract2(self, feat):
        if self.lam is not None and self.training:
            mrs = torch.cat([layer.get_mr()
                             for layer in self.fuzzylayers_C.values()], dim=0)
            self.loss_p_size = self.inner_num
            return self.lam * (- torch.sum(mrs * torch.log(mrs))), self.abstract(feat)
        else:
            self.loss_p_size = self.nkshot + self.nkquery
            return 0., self.abstract(feat)


class FWeDLEO(FWeDParall):
    def __init__(self, param_size, weight_size, Rule, len_rules, set_num, inner_num,
                 inner_lr, inner_wd, match, layers, param_enc):
        super(FWeDLEO, self).__init__(param_size, weight_size, Rule, len_rules, set_num, inner_num,
                                      inner_lr, inner_wd, match, layers, init_param=False)
        self.param_enc = param_enc
        if param_size % param_enc.hidden_size:
            raise ValueError('It must be a integer time between param_size and param_enc.hidden_size!')
        self.cte = ChTimes(param_enc.channels, self.len_fl, param_size // param_enc.hidden_size)

    def init_param_inner(self, num):
        self.param_inner = self.cte(self.param_enc(self.feat, self.label, self.featb).transpose(-2, -1).contiguous()
                                    ).transpose(-2, -1).contiguous().view(-1, self.param_size)

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat = self.nonfinetune(torch.cat([feat, self.featb], dim=0))
            feat = feat if self.training else feat.detach()
            self.featb = feat[self.nkshot:]
            return feat[:self.nkshot]
        else:
            return feat

    def abstract1(self, feat):
        return 0., self.abstract(feat)

    def inner_fit(self, feata, label, featb):
        self.featb = featb.transpose(0, 1).contiguous()
        return super().inner_fit(feata, label)

    def inner_test(self, label, classify=False, reduce=True, gagree=False):
        self.rules.inner_eval()
        feat = self.featb
        self.label_size = list(label.size())
        self.label_size = [self.label_size[0] * self.label_size[1]] + self.label_size[2:]
        self.label = label.transpose(0, 1).contiguous()
        label = self.label.view(*self.label_size)
        self.loss_p = 0.
        self.create_weights(self.num)
        self.net_C[0].inner_eval()

        output = self.net_C[0](feat)
        output_size = list(output.size())
        output_size = [output_size[0] * output_size[1]] + output_size[2:]
        if reduce:
            return self.num * self.lossf(output.view(*output_size), label), 0.
        else:
            loss = self.lossf(output.view(*output_size), label, reduction='none')
            loss = loss.view(-1, self.num)
            if gagree:
                return loss.mean(0), 0.
            elif classify:
                self.label = self.label.view(self.label_size[0] // self.num, self.num)
                c = (torch.argmax(output, 2) == self.label).type(FloatTensor)
                return loss.mean(0), c.mean(0)
            else:
                return loss.mean(0)

    def forward(self, feata, labela, featb):
        self.inner_fit(feata, labela, featb)
        feat_size = featb.size()
        self.num = feat_size[0]
        self.nkquery = feat_size[1]
        self.create_weights(self.num)
        self.net_C[0].inner_eval()
        self.rules.inner_eval()
        return self.net_C[0](self.featb).transpose(0, 1).contiguous()


class FCoAParall(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(FCoAParall, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        # inner learning rate
        self.lr_pw = Parameter(torch.zeros(self.len_w))
        self.lr_pw.data.fill_(self.inner_lr)
        self.lr_pfl = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pfl.data.fill_(self.inner_lr)

        # inner cosine annealing
        self.lr_pa = Parameter(torch.zeros(self.len_w))
        self.lr_pa.data.fill_(1e-4)
        self.lr_pfla = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pfla.data.fill_(1e-4)

        # inner weight decay
        self.inner_wd = inner_wd
        self.lr_pwed = Parameter(torch.zeros(self.len_w))
        self.lr_pwed.data.fill_(self.inner_lr * self.inner_wd)
        self.lr_pflwed = Parameter(torch.zeros(self.len_fl, 1))
        self.lr_pflwed.data.fill_(self.inner_lr * self.inner_wd)

    def init_inner(self, num):
        super().init_inner(num)
        self.num = num
        self.lr_pflwed_inner = self.lr_pflwed.clone().repeat(num, 1)
        self.step = 0.

    def inner_update(self, grads):
        lr_p = self.lr_pw / 2 * (torch.cos(self.lr_pa * self.step) + 1)
        lr_pfl_inner = (self.lr_pfl / 2 * (torch.cos(self.lr_pfla * self.step) + 1)).repeat(self.num, 1)
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (lr_p[i] * g + self.lr_pwed[i] * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - (
            lr_pfl_inner * grads[-1] + self.lr_pflwed_inner * self.param_inner)
        self.step += 1.

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class F1WeD1RL(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, inner_wd, match, layers, lam=None, rho=0.2):
        super(F1WeD1RL, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        self.inner_wd = inner_wd
        self.inner_lrP = Parameter(torch.Tensor([np.sqrt(inner_lr)]))
        self.inner_wdP = Parameter(torch.Tensor([np.sqrt(inner_wd)]))

    def inner_update(self, grads):
        #print('inner_lrP=', self.inner_lrP ** 2)
        #print('inner_wdP=', self.inner_wdP ** 2)
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            m[wn] = m[wn] - (self.inner_lrP ** 2) * (g + (self.inner_wdP ** 2) * m[wn])
            self.use_wlist[i] = m[wn]
        self.param_inner = self.param_inner - self.inner_lrP * (
            grads[-1] + self.inner_wdP * self.param_inner)

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, inner_num={inner_num}, '
             'inner_wd={inner_wd}, param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class FNetEncode(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, decode, lam=None, rho=0.2):
        super(FNetEncode, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        self.decode = decode

    def abstract(self, feat):
        if self.nonfinetune is not None:
            feat1 = self.nonfinetune(feat)
            if self.lam is not None:
                loss_p = self.lam * F.mse_loss(
                    feat, self.decode(feat1), reduction='sum')
            else:
                loss_p = 0.
            return loss_p, feat1
        else:
            return 0., feat


class FNetProto(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam=None, rho=0.2):
        super(FNetProto, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers, lam, rho)
        self.alpha = Parameter(torch.Tensor([1.]))

    def abstract1(self, feat):
        if self.nonfinetune is not None:
            if self.lam is not None:
                self.nway = len(torch.unique(self.label, sorted=True))
                if self.nkshot % self.nway != 0:
                    raise ValueError("The number of shot in each way should be equal!")
                else:
                    self.kshot = self.nkshot // self.nway
                self.abstract1 = self.proto1
                return self.proto1(feat)
        return 0., F.relu(self.nonfinetune(feat))

    def proto1(self, feat):
        feat = self.nonfinetune(feat)
        self.elems = torch.unique(self.label, sorted=True)
        self.support = torch.cat([feat[self.label == elem].view(self.kshot, self.num, 1, -1)
                                  for elem in self.elems], dim=2).mean(dim=0, keepdim=True)
        return 0., feat

    def abstract2(self, feat):
        feat = self.nonfinetune(feat)
        query = feat.view(self.nkquery, self.num, 1, -1)
        dis = self.alpha * torch.sum((query - self.support) ** 2, dim=3)
        dis_size = [self.nkquery * self.num, self.nway]
        self.loss_p_size = self.nkquery
        return self.lam * F.cross_entropy(dis.view(*dis_size), self.label.view(
            *self.label_size), reduction='sum'), feat


class FSGDParall(_MetaBase):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers):
        super(FSGDParall, self).__init__()
        self.net = nn.Sequential(*layers)
        self.len_rules = len_rules
        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.match = match
        self.net_C = [nn.Sequential()]
        self.len_fl = 0
        self.weightn_list = []
        self.fuzzylayers = OrderedDict()
        self.wnames = []
        self.wrs = []
        self.lr_pw = []
        self.mdict = []
        self.bws = []

        for i, module in enumerate(self.net._modules.values()):
            if isinstance(module, FuzzyBase):
                self.fuzzylayers[str(i)] = module
                module.get_setnum(self.len_rules, self.match)
                self.len_fl += module.num_weight
                self.weightn_list.append(module.num_weight)

            moduleparall = self.to_parall(module)
            self.net_C[0].add_module(str(i), moduleparall)

            if isinstance(module, FuzzyBase):
                self.fuzzylayers_C[str(i)] = moduleparall

            if isinstance(module, FCh2D) and isinstance(module.bw, BatchWeight):
                self.bws.append(module.bw)

            if isinstance(module, FCh2D):
                continue
            for name, param in module.named_parameters():
                self.mdict.append(moduleparall)
                self.wnames.append(name)
                self.wrs.append(getattr(moduleparall, name + '_r'))
                setattr(self,str(i)+name, Parameter(torch.ones(param.size())))
                self.lr_pw.append(getattr(self, str(i)+name))

        self.len_w = len(self.wnames)
        self.use_wlist = [[]] * (self.len_w + 1)
        self.mdict = tuple(self.mdict)
        self.wnames = tuple(self.wnames)
        self.wrs = tuple(self.wrs)
        self.lr_pw = tuple(self.lr_pw)

        if 2 ** set_num != self.len_rules:
            raise Exception('The number of rules is not equal to the number of divide regions.')

        self.param_size = param_size
        if isinstance(weight_size, int):
            self.weight_size = weight_size
        else:
            weight_size = (weight_size[1], weight_size[0])
            self.weight_size = weight_size

        self.rules = Rule(len_rules, param_size, weight_size, inner_num, self.len_fl)

        for i, module in enumerate(self.rules._modules.values()):
            if isinstance(module, BatchWeight):
                self.bws.append(module)
        self.bws = tuple(self.bws)

        self.param = Parameter(torch.Tensor(self.len_fl, param_size))
        self.lr_pfl = Parameter(torch.ones(self.len_fl, param_size))
        self.reset_parameters()
        if isinstance(self.match, EachMatch):
            self.match.init_params()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(sum(self.param.size()) / self.len_fl)
        self.param.data.uniform_(-stdv, stdv)

    def init_inner(self, num):
        super().init_inner(num)
        for bw in self.bws:
            bw.reset()

        #self.param_inner = self.param / torch.norm(self.param, p=2)
        self.param_inner = self.param.clone()
        self.param_inner = self.param_inner.repeat(num, 1)
        self.lr_pfl_inner = self.lr_pfl.clone().repeat(num, 1)

        self.lr_pw_inner = []
        for lr, wr in zip(self.lr_pw, self.wrs):
            self.lr_pw_inner.append(wr(lr))

    def create_weights(self, num):
        self.use_wlist[-1] = self.param_inner

        self.weights = self.rules(self.param_inner).view(
            num, self.len_fl, self.len_rules, self.weight_size[0], self.weight_size[1])

        index = 0
        for weightn, layer in zip(self.weightn_list, self.fuzzylayers_C.values()):
            layer.get_weights(self.weights[:, index:index + weightn])
            index += weightn

    def inner_update(self, grads):
        for i, g, m, wn in zip(range(self.len_w), grads[:-1], self.mdict, self.wnames):
            h = m.__dict__[wn] - self.inner_lr * self.lr_pw_inner[i] * g
            setattr(m, wn, h)
            self.use_wlist[i] = h
        self.param_inner = self.param_inner - self.inner_lr * self.lr_pfl_inner * grads[-1]
        self.use_wlist[-1] = self.param_inner

    def extra_repr(self):
        s = ('len_rules={len_rules}, inner_lr={inner_lr}, '
             'param_size={param_size}')
        if isinstance(self.weight_size, int):
            s += ', weight_size=' + str(self.weight_size)
        else:
            s += ', weight_size=' + str((self.weight_size[1], self.weight_size[0]))
        return s.format(**self.__dict__)


class FuzzyNetOL(FNetParall):
    def __init__(self, param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers):
        super(FuzzyNetOL, self).__init__(param_size, weight_size, Rule, len_rules,
                 set_num, inner_num, inner_lr, match, layers)
        for m in self.fuzzylayers_C.values():
            if isinstance(m, FFCNN2DParall) or isinstance(m, FCNN2DParall):
                m.get_rules(self.rules)

    def create_weights(self, num):
        self.param_inner = self.param.clone()
        self.param_inner = self.param_inner.repeat(num, 1)
        self.use_wlist[-1] = self.param_inner

        self.param_temp = self.param_inner.view(num, self.len_fl, -1)

        index = 0
        for weightn, layer in zip(self.weightn_list, self.fuzzylayers_C.values()):
            layer.get_weights(self.param_temp[:, index:index + weightn])
            index += weightn


class NochoseBase(nn.Module):
    def __init__(self):
        super(NochoseBase, self).__init__()
        self.inner_training = False

    def extra_repr(self):
        return '{input_size}, {output_size}, in_channels={in_channels}, ' \
               'out_channels={out_channels}'.format(**self.__dict__)

    def get_setnum(self, len_rules):
        self.len_rules = len_rules
        self.set_num = np.int(np.log2(len_rules))
        self.mr = torch.ones(self.rule_channels, len_rules).type(FloatTensor)

    def get_weights(self, weight):
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        mr = self.mr

        weight = torch.sum(self.weight * mr.repeat(
            self.out_channels, 1).view(self.num_weight, self.len_rules, 1, 1), 1)
        inw_size = weight.size()
        weight = weight.view(self.out_channels, self.in_channels, inw_size[-2], inw_size[-1])
        weight = F.upsample(weight, self.weight_size, mode='bilinear', align_corners=True)
        weight = weight.unsqueeze(0)
        x = x.view(batch_size, 1, self.in_channels, 1, self.weight_size[1])
        return torch.sum(torch.matmul(x, weight.transpose(-2, -1)), 2).view(
            batch_size, self.out_channels, self.output_size[0], self.output_size[1])


class NochoseParall(ModuleParall):
    def __init__(self):
        super(NochoseParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.input_size = model.input_size
        self.output_size = model.output_size
        self.weight_size = model.weight_size
        self.in_channels = model.in_channels
        self.rule_channels = model.rule_channels
        self.out_channels = model.out_channels
        self.num_weight = model.num_weight
        self.training = model.training
        self.inner_training = model.inner_training

        self.len_rules = model.len_rules
        self.set_num = model.set_num
        self.mr = model.mr.unsqueeze(0).repeat(num,1,1)
        self.match = model.match

        self.num = num

    def get_weights(self, weight):
        self.weight = weight

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        mr = self.mr

        weight = torch.sum(self.weight * mr.repeat(
            1, self.out_channels, 1).view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        inw_size = weight.size()
        weight = weight.view(self.num * self.out_channels, self.in_channels, inw_size[-2], inw_size[-1])
        if (inw_size[-2] != self.weight_size[0]) or (inw_size[-1] != self.weight_size[0]):
            weight = F.upsample(weight, self.weight_size, mode='bilinear', align_corners=True)
        weight = weight.view(1, self.num, self.out_channels, self.in_channels, self.weight_size[0], self.weight_size[1])
        x = x.view(batch_size, self.num, 1, self.in_channels, 1, self.weight_size[1])
        return torch.sum(torch.matmul(x, weight.transpose(-2, -1)), 3).view(
            batch_size, self.num, self.out_channels, self.output_size[0], self.output_size[1])


class NCh2D(NochoseBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(NCh2D, self).__init__()
        self.in_channels = in_channels
        self.rule_channels = in_channels
        self.out_channels = out_channels
        self.num_weight = in_channels * out_channels
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

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.num_weight = self.rule_channels = int(np.prod(self.kernel_size))
        self.bw = InstanceNormWeight(self.num_weight, affine=True)

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
        return s.format(**self.__dict__)

    def get_setnum(self, len_rules):
        super().get_setnum(len_rules)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param


class NCh2DParall(NochoseParall):
    def __init__(self):
        super(NCh2DParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.in_channels = model.in_channels
        self.rule_channels = model.rule_channels
        self.out_channels = model.out_channels
        self.num_weight = model.num_weight
        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation = model.dilation
        self.inner_training = model.inner_training
        self.len_rules = model.len_rules
        self.set_num = model.set_num
        self.num = num

        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias.repeat(num)

        self.bw = model.bw
        self.mr = model.mr.unsqueeze(0).repeat(num, 1, 1)

    def get_rules(self, rules):
        raise ValueError('The get_rules is unused in NCh2D!')

    def forward(self, x):
        mr = self.mr

        weight = torch.sum(self.weight * mr.view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        weight = self.bw(weight)
        inw_size = weight.size()
        if (inw_size[-2] != self.out_channels) or (inw_size[-1] != self.in_channels):
            weight = F.interpolate(weight, (self.out_channels, self.in_channels), mode='bilinear', align_corners=True)
        weight = weight.permute(0,2,3,1).reshape(self.num * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


class FuzzyBase(nn.Module):
    def __init__(self):
        super(FuzzyBase, self).__init__()
        self.inner_training = False

    def get_setnum(self, len_rules, match):
        self.len_rules = len_rules
        self.set_num = np.int(np.log2(len_rules))
        self.mr = torch.zeros(self.rule_channels, len_rules).type(FloatTensor)
        self.match = (match,)

    def get_weights(self, weight):
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        dim1 = batch_size * self.rule_channels
        if self.inner_training:
            '''
            mr1 = F.hardtanh(self.match[0](x.view(dim1, 1, self.input_size[0], self.input_size[1])
                                           ).view(batch_size, self.rule_channels, 1, self.set_num).mean(0)) / 2. + 0.5
            dim1 = self.in_channals
            '''
            mr1 = self.match[0](x)

            mr2 = 1. - mr1
            mr = Variable(torch.zeros(dim1, self.len_rules, self.set_num)).type(FloatTensor)
            mr[:, :self.len_rules // 2] = mr1
            mr[:, self.len_rules // 2:] = mr2

            h = 1
            for i in range(1, self.set_num):
                h *= 2
                mri = mr[:, :, i].view(dim1, -1, h).transpose(1, 2)
                mr[:, :, i] = mri.reshape(dim1, self.len_rules)
            mr = mr.prod(2)
            self.mr = (1 - self.momentum) * self.mr + self.momentum
        else:
            mr = self.mr

        return self.step2(x, mr, batch_size)


class Fuzzy2D(FuzzyBase):
    def __init__(self, input_size, output_size, in_channels, out_channels):
        super(Fuzzy2D, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight_size = (np.prod(output_size), np.prod(input_size))
        self.in_channels = in_channels
        self.rule_channels = in_channels
        self.out_channels = out_channels
        self.num_weight = in_channels * out_channels

    def extra_repr(self):
        return '{input_size}, {output_size}, in_channels={in_channels}, ' \
               'out_channels={out_channels}'.format(**self.__dict__)

    def step2(self, x, mr, batch_size):
        weight = torch.sum(self.weight * mr.repeat(
            self.out_channels, 1).view(self.num_weight, self.len_rules, 1, 1), 1)
        inw_size = weight.size()
        weight = weight.view(self.out_channels, self.in_channels, inw_size[-2], inw_size[-1])
        weight = F.upsample(weight, self.weight_size, mode='bilinear', align_corners=True)
        weight = weight.unsqueeze(0)
        x = x.view(batch_size, 1, self.in_channels, 1, self.weight_size[1])
        return torch.sum(torch.matmul(x, weight.transpose(-2, -1)), 2).view(
            batch_size, self.out_channels, self.output_size[0], self.output_size[1])

'''
def main():
    x = Variable(torch.randn(24*4).view(4, 2, 3, 2, 2)).cuda()
    xt = x.transpose(0,1).contiguous()
    match = nn.Conv2d(1,3,2).cuda()
    w = Variable(torch.arange(0, 9*8*16).view(9, 8, 4, 4)).cuda()
    fn = Fuzzy2D((2, 2), (2, 2), 3, 3)
    fn.get_setnum(8,match)
    fn.get_weights(w)
    fns = Fuzzy2DParall()
    fns.get_parameters(fn,4)
    fns.get_weights(w.unsqueeze(0).repeat(4,1,1,1,1))
    k = 1
    print(fns(xt).transpose(0,1).contiguous()[k])
    print([fn(x[i])for i in range(4)][k])
'''

class Fuzzy2DParall(ModuleParall):
    def __init__(self):
        super(Fuzzy2DParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.input_size = model.input_size
        self.output_size = model.output_size
        self.weight_size = model.weight_size
        self.in_channels = model.in_channels
        self.rule_channels = model.rule_channels
        self.out_channels = model.out_channels
        self.num_weight = model.num_weight
        self.training = model.training
        self.inner_training = model.inner_training

        self.len_rules = model.len_rules
        self.set_num = model.set_num
        self.mr = model.mr.unsqueeze(0).repeat(num,1,1)
        self.match = model.match

        self.num = num

    def get_weights(self, weight):
        self.weight = weight

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        '''
        dim1 = batch_size * self.num * self.in_channels
        dim2 = self.num * self.in_channels
        '''
        dim1 = self.num * self.rule_channels
        if self.inner_training:
            '''
            mr1 = F.hardtanh(self.match[0](x.view(dim1, 1, input_size[3], input_size[4])
                                           ).view(batch_size, dim2, 1, self.set_num).mean(0)) / 2. + 0.5
            dim1 = dim2
            '''
            mr1 = self.match[0](x)

            mr2 = 1. - mr1
            mr = Variable(torch.zeros(dim1, self.len_rules, self.set_num)).type(FloatTensor)
            mr[:, :self.len_rules // 2] = mr1
            mr[:, self.len_rules // 2:] = mr2

            h = 1
            for i in range(1, self.set_num):
                h *= 2
                mri = mr[:, :, i].view(dim1, -1, h).transpose(1,2)
                mr[:,:,i] = mri.reshape(dim1, self.len_rules)
            mr = mr.prod(2).view(self.num, self.rule_channels, self.len_rules)
            self.mr = mr
        else:
            mr = self.mr

        return self.step2(x, mr, batch_size)

    def step2(self, x, mr, batch_size):
        weight = torch.sum(self.weight * mr.repeat(
            1, self.out_channels, 1).view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        inw_size = weight.size()
        weight = weight.view(self.num * self.out_channels, self.in_channels, inw_size[-2], inw_size[-1])
        if (inw_size[-2] != self.weight_size[0]) or (inw_size[-1] != self.weight_size[0]):
            weight = F.upsample(weight, self.weight_size, mode='bilinear', align_corners=True)
        weight = weight.view(1, self.num, self.out_channels, self.in_channels, self.weight_size[0], self.weight_size[1])
        x = x.view(batch_size, self.num, 1, self.in_channels, 1, self.weight_size[1])
        return torch.sum(torch.matmul(x, weight.transpose(-2, -1)), 3).view(
            batch_size, self.num, self.out_channels, self.output_size[0], self.output_size[1])


class FFCNN2D(Fuzzy2D):
    def __init__(self, input_size, output_size, in_channals, out_channals, stride, row):
        super(FFCNN2D, self).__init__(input_size, output_size, in_channals, out_channals)
        #self.stride = (stride[0] * input_size[1], stride[1])
        self.stride = stride
        self.row = row

    def step2(self, x, mr, batch_size):
        pass


class FFCNN2DParall(Fuzzy2DParall):
    def __init__(self):
        super(FFCNN2DParall, self).__init__()

    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)
        self.stride = model.stride
        self.row = model.row

    def get_rules(self, rules):
        self.rules = (rules,)

    def step2(self, x, mr, batch_size):
        weight1 = self.rules[0](self.weight.reshape(self.num * self.num_weight, -1)).view(
            self.num, self.num_weight, self.len_rules, -1)

        weight1 = torch.sum(weight1 * mr.repeat(
            1, self.out_channals, 1).view(self.num, self.num_weight, self.len_rules, 1), 2)
        inw_size = weight1.size()
        weight1 = weight1.view(self.num * self.out_channals, self.in_channals, inw_size[-1])
        weight1 = F.upsample(weight1, int(self.weight_size[1]), mode='linear', align_corners=True)

        '''
        calcu_size = [1, self.num, -1, self.in_channals * self.weight_size[1]]
        weight1 = weight1.view(self.num, self.out_channals, self.in_channals, self.weight_size[1])
        out_size = [batch_size, self.num, self.out_channals, -1]
        x = x.view(batch_size, self.num, 1, self.in_channals * self.weight_size[1])
        xs = []

        weight11 = weight1
        weight = [weight1]
        k = 1
        if k % self.row == 0:
            k = 0
            weight = torch.cat(weight, 1)
            xs.append(x.matmul(weight.view(*calcu_size).transpose(-2, -1)).view(*out_size))
            weight = []

        for _ in range(self.output_size[1] - 1):
            weight1 = torch.cat((weight1[:, :, :, self.stride[1]:],
                                 weight1[:, :, :, :self.stride[1]]), 3)
            weight.append(weight1)
            k += 1
            if k % self.row == 0:
                k = 0
                weight = torch.cat(weight, 1)
                xs.append(x.matmul(weight.view(*calcu_size).transpose(-2, -1)).view(*out_size))
                weight = []

        for _ in range(self.output_size[0] - 1):
            weight11 = torch.cat((weight11[:, :, :, self.stride[0]:],
                                  weight11[:, :, :, :self.stride[0]]), 3)
            weight.append(weight11)
            weight1 = weight11
            k += 1
            if k % self.row == 0:
                k = 0
                weight = torch.cat(weight, 1)
                xs.append(x.matmul(weight.view(*calcu_size).transpose(-2, -1)).view(*out_size))
                weight = []

            for _ in range(self.output_size[1] - 1):
                weight1 = torch.cat((weight1[:, :, :, self.stride[1]:],
                                     weight1[:, :, :, :self.stride[1]]), 3)
                weight.append(weight1)
                k += 1
                if k % self.row == 0:
                    k = 0
                    weight = torch.cat(weight, 1)
                    xs.append(x.matmul(weight.view(*calcu_size).transpose(-2, -1)).view(*out_size))
                    weight = []

        if k != 0:
            weight = torch.cat(weight, 1)
            xs.append(x.matmul(weight.view(*calcu_size).transpose(-2, -1)).view(*out_size))

        return torch.cat(xs, 3).view(
            batch_size, self.num, self.out_channals, self.output_size[0], self.output_size[1])
        '''

        weight1 = weight1.view(self.num * self.out_channals, self.in_channals, self.input_size[0], self.input_size[1])
        weight = weight1.repeat(1, 1, 2, 2)
        weight = weight[:, :, :-1, :] if self.stride[0] == 1 else weight
        weight = weight[:, :, :, :-1] if self.stride[1] == 1 else weight
        x = x.view(batch_size, self.num * self.in_channals, self.input_size[0], self.input_size[1])
        return F.conv2d(x, weight, None, self.stride, (
            self.input_size[0]-1, self.input_size[1]-1), 1, self.num).view(
            batch_size, self.num, self.out_channals, self.output_size[0], self.output_size[1])


class FCNN2D(FuzzyBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(FCNN2D, self).__init__()
        self.in_channels = in_channels
        self.rule_channels = in_channels
        self.out_channels = out_channels
        self.num_weight = in_channels * out_channels
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
        return s.format(**self.__dict__)


class FCNN2DParall(Fuzzy2DParall):
    def __init__(self):
        super(FCNN2DParall, self).__init__()

    def get_parameters(self, model, num=1):
        self.in_channels = model.in_channels
        self.rule_channels = model.rule_channels
        self.out_channels = model.out_channels
        self.num_weight = model.num_weight
        self.kernel_size = model.kernel_size
        self.stride = model.stride
        self.padding = model.padding
        self.dilation = model.dilation
        self.inner_training = model.inner_training
        self.len_rules = model.len_rules
        self.set_num = model.set_num
        self.match = model.match
        self.num = num

        if model.bias is None:
            self.bias = None
        else:
            self.bias = model.bias.clone()
            self.bias = self.bias.repeat(num)

        self.get_param_option(model, num)

    def get_param_option(self, model, num):
        self.mr = model.mr.unsqueeze(0).repeat(num, 1, 1)

    def get_rules(self, rules):
        self.rules = (rules,)

    def step2(self, x, mr, batch_size):
        weight1 = self.rules[0](self.weight.reshape(self.num * self.num_weight, -1)).view(
            self.num, self.num_weight, self.len_rules, -1)

        weight1 = torch.sum(weight1 * mr.repeat(
            1, self.out_channels, 1).view(self.num, self.num_weight, self.len_rules, 1), 2)
        weight1 = weight1.view(self.num * self.out_channels, self.in_channels, -1)
        weight1 = F.upsample(weight1, self.kernel_size[0] * self.kernel_size[1],
                             mode='linear', align_corners=True)
        weight1 = weight1.view(self.num * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight1, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


class InstanceNormWeight(_InstanceNorm):
    def _check_input_dim(self, input):
        pass
    def inner_train(self, mode=True):
        return self
    def inner_eval(self):
        return self


class FCh2D(FCNN2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(FCh2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias)
        self.num_weight = self.rule_channels = int(np.prod(self.kernel_size))
        self.bw = InstanceNormWeight(self.num_weight, affine=True)

    def get_setnum(self, len_rules, match):
        super().get_setnum(len_rules, match)
        self.match = (CapChMatchShell(match, self.kernel_size,
                                      self.stride, self.padding, self.dilation),)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param


class FCh2DParall(FCNN2DParall):
    def __init__(self):
        super(FCh2DParall, self).__init__()

    def get_param_option(self, model, num):
        self.bw = model.bw
        self.mr = model.mr.unsqueeze(0).repeat(num, 1, 1)

    def get_rules(self, rules):
        raise ValueError('The get_rules is unused in FCh2D!')

    def step2(self, x, mr, batch_size):
        weight = torch.sum(self.weight * mr.view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        weight = self.bw(weight)
        inw_size = weight.size()
        if (inw_size[-2] != self.out_channels) or (inw_size[-1] != self.in_channels):
            weight = F.interpolate(weight, (self.out_channels, self.in_channels), mode='bilinear', align_corners=True)
        weight = weight.permute(0,2,3,1).reshape(self.num * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


class F2DEach(FCNN2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(F2DEach, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias)
        self.num_weight = self.rule_channels = int(np.prod(self.kernel_size))
        self.bw = InstanceNormWeight(self.num_weight, affine=True)

    def get_setnum(self, len_rules, match):
        self.len_rules = len_rules
        self.set_num = np.int(np.log2(len_rules))
        self.site = match.add_field(self.num_weight)
        self.match = (match,)


class F2DEachParall(FCNN2DParall):
    def __init__(self):
        super(F2DEachParall, self).__init__()

    def get_param_option(self, model, num):
        self.bw = model.bw
        self.site = model.site

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        if self.inner_training:
            mr1 = self.match[0](self.site)

            mr2 = 1. - mr1
            mr = Variable(torch.zeros(self.num_weight, self.len_rules, self.set_num)).type(FloatTensor)
            mr[:, :self.len_rules // 2] = mr1
            mr[:, self.len_rules // 2:] = mr2

            h = 1
            for i in range(1, self.set_num):
                h *= 2
                mri = mr[:, :, i].view(self.num_weight, -1, h).transpose(1,2)
                mr[:,:,i] = mri.reshape(self.num_weight, self.len_rules)
            mr = mr.prod(2).view(1, self.num_weight, self.len_rules).expand(self.num, -1, -1)
            self.mr = mr
        else:
            mr = self.mr

        return self.step2(x, mr, batch_size)

    def step2(self, x, mr, batch_size):
        weight = torch.sum(self.weight * mr.view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        weight = self.bw(weight)
        inw_size = weight.size()
        if (inw_size[-2] != self.out_channels) or (inw_size[-1] != self.in_channels):
            weight = F.interpolate(weight, (self.out_channels, self.in_channels), mode='bilinear', align_corners=True)
        weight = weight.permute(0,2,3,1).reshape(self.num * self.out_channels, self.in_channels,
                                                 self.kernel_size[0], self.kernel_size[1])

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


# class FCh2DE(FCh2D):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, bias=True):
#         super(FCh2DE, self).__init__(in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, bias)
#
#     def get_setnum(self, len_rules, match):
#         super().get_setnum(len_rules, match)
#         self.match = (match,)
#
#
# class FCh2DEParall(FCh2DParall):
#     def __init__(self):
#         super(FCh2DEParall, self).__init__()


class FCh2DSP(FCh2D):
    pass


class FCh2DSPParall(FCh2DParall):
    def __init__(self):
        super(FCh2DSPParall, self).__init__()
        self.mrs = []

    def step2(self, x, mr, batch_size):
        if self.training and self.inner_training:
            self.mrs.append(mr)
        return super().step2(x, mr, batch_size)

    def get_mr(self):
        mrs = torch.cat([mr.view(-1, self.len_rules) for mr in self.mrs], dim=0)
        self.mrs = []
        return mrs


class FCh2D1M(FCh2D):
    pass


class FCh2D1MParall(FCh2DParall):
    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)
        self.match_able = True

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        dim1 = self.num * self.rule_channels
        if self.match_able:
            mr1 = self.match[0](x)

            mr2 = 1. - mr1
            mr = Variable(torch.zeros(dim1, self.len_rules, self.set_num)).type(FloatTensor)
            mr[:, :self.len_rules // 2] = mr1
            mr[:, self.len_rules // 2:] = mr2

            h = 1
            for i in range(1, self.set_num):
                h *= 2
                mri = mr[:, :, i].view(dim1, -1, h).transpose(1,2)
                mr[:,:,i] = mri.reshape(dim1, self.len_rules)
            mr = mr.prod(2).view(self.num, self.rule_channels, self.len_rules)
            self.mr = mr
            self.match_able = False
        else:
            mr = self.mr

        return self.step2(x, mr, batch_size)


class FCh2DAll(FCh2D):
    pass


class FCh2DAllParall(FCh2DParall):
    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        dim1 = self.num * self.rule_channels
        mr1 = self.match[0](x)

        mr2 = 1. - mr1
        mr = Variable(torch.zeros(dim1, self.len_rules, self.set_num)).type(FloatTensor)
        mr[:, :self.len_rules // 2] = mr1
        mr[:, self.len_rules // 2:] = mr2

        h = 1
        for i in range(1, self.set_num):
            h *= 2
            mri = mr[:, :, i].view(dim1, -1, h).transpose(1,2)
            mr[:,:,i] = mri.reshape(dim1, self.len_rules)
        mr = mr.prod(2).view(self.num, self.rule_channels, self.len_rules)

        return self.step2(x, mr, batch_size)


class FCh2DMH(FCh2D):
    pass


class FCh2DMHParall(FCh2DParall):
    def get_parameters(self, model, num=1):
        super().get_parameters(model, num)
        self.match_able = True

    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        dim1 = self.num * self.rule_channels
        if self.match_able:
            mr1 = self.match[0](x)

            mr2 = 1. - mr1
            mr = Variable(torch.zeros(dim1, self.len_rules, self.set_num)).type(FloatTensor)
            mr[:, :self.len_rules // 2] = mr1
            mr[:, self.len_rules // 2:] = mr2

            h = 1
            for i in range(1, self.set_num):
                h *= 2
                mri = mr[:, :, i].view(dim1, -1, h).transpose(1,2)
                mr[:,:,i] = mri.reshape(dim1, self.len_rules)
            mr = mr.prod(2).view(self.num, self.rule_channels, self.len_rules)
        else:
            mr = None

        return self.step2(x, mr, batch_size)

    def step2(self, x, mr, batch_size):
        if self.training:
            self.weight_u = torch.sum(self.weight * mr.view(self.num, self.num_weight, self.len_rules, 1, 1), 2)
        else:
            self.match_able = False
        weight = self.weight_u
        weight = self.bw(weight)
        inw_size = weight.size()
        if (inw_size[-2] != self.out_channels) or (inw_size[-1] != self.in_channels):
            weight = F.interpolate(weight, (self.out_channels, self.in_channels), mode='bilinear', align_corners=True)
        weight = weight.permute(0,2,3,1).reshape(self.num * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        si = x.size()
        x = x.view(si[0], si[1] * si[2], si[3], si[4])
        output = F.conv2d(x, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.num)
        so = output.size()[-2:]
        return output.view(si[0], si[1], self.out_channels, so[0], so[1])


class FuzzyBlock(nn.Module):
    pass


class FuzzyBlock2D(FuzzyBlock):
    expansion = 1
    def __init__(self, block, inplanes, planes, stride=1, downsample=None):
        super(FuzzyBlock2D, self).__init__()
        self.conv1 = block(inplanes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.conv2 = block(planes, planes, kernel_size=3, padding=1, bias=False)
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


class FuzzyBlock2DParall(ModuleParall):
    expansion = 1
    def __init__(self, m):
        super(FuzzyBlock2DParall, self).__init__()
        self.conv1 = to_parall(m.conv1)
        self.bn1 = BatchNormParall()
        self.relu = nn.ReLU()
        self.conv2 = to_parall(m.conv2)

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


class CNNFuzzyBlock2D(FuzzyBlock2D):
    expansion = 1

    def __init__(self, block, inplanes, planes, stride=1, downsample=None):
        super(FuzzyBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.conv2 = block(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)

        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion, track_running_stats=False),
            )
        self.downsample = downsample
        self.stride = stride


class Fuzzy2D333(FuzzyBlock):
    expansion = 1
    def __init__(self, block, inplanes, planes, stride=1, downsample=None):
        super(Fuzzy2D333, self).__init__()
        self.conv1 = block(inplanes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.conv2 = block(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)
        self.conv3 = block(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, affine=True, track_running_stats=False)

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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Fuzzy2D333Parall(ModuleParall):
    expansion = 1
    def __init__(self, m):
        super(Fuzzy2D333Parall, self).__init__()
        self.conv1 = to_parall(m.conv1)
        self.bn1 = BatchNormParall()
        self.relu = nn.ReLU()
        self.conv2 = to_parall(m.conv2)
        self.bn2 = BatchNormParall()
        self.conv3 = to_parall(m.conv3)
        self.bn3 = BatchNormParall()

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
        self.conv3.get_parameters(model.conv3, num)
        self.bn3.get_parameters(model.bn3, num)
        if self.downsample is not None:
            self.downsample.get_parameters(model.downsample, num)

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


class FuzzyDense2D(FuzzyBlock):

    def __init__(self, block, inplanes, growth_rate, bn_size=4):
        super(FuzzyDense2D, self).__init__()
        self.conv1 = block(inplanes, growth_rate * bn_size, kernel_size=3, stride=1,
                           padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate * bn_size, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.conv2 = block(growth_rate * bn_size, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate, affine=True, track_running_stats=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = torch.cat([x, self.relu(self.bn2(out))], 1)
        return out


class FuzzyDense2DParall(ModuleParall):
    def __init__(self, m):
        super(FuzzyDense2DParall, self).__init__()
        self.conv1 = to_parall(m.conv1)
        self.bn1 = BatchNormParall()
        self.relu = nn.ReLU()
        self.conv2 = to_parall(m.conv2)
        self.bn2 = BatchNormParall()

    def get_parameters(self, model, num=1):
        self.conv1.get_parameters(model.conv1, num)
        self.bn1.get_parameters(model.bn1, num)
        self.conv2.get_parameters(model.conv2, num)
        self.bn2.get_parameters(model.bn2, num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = torch.cat([x, self.relu(self.bn2(out))], 2)
        return out

class RInception(nn.Module):
    def __init__(self, param_size, weight_size):
        super(RInception, self).__init__()
        self.weight_size = np.array(weight_size)
        self.param_size = np.array(param_size)

        ks1 = 1
        ks3 = 3

        st3 = np.ceil((self.weight_size - ks3) / (self.param_size - 1)).astype(np.int)
        pad3 = (self.param_size - 1) * st3 - self.weight_size + ks3
        output_pad3 = pad3 % 2
        pad3 = ((pad3 + output_pad3) / 2).astype(np.int)
        st3 = tuple(st3)
        pad3 = tuple(pad3)
        output_pad3 = tuple(output_pad3)

        st1 = np.ceil((self.weight_size - ks1) / (self.param_size - 1)).astype(np.int)
        pad1 = (self.param_size - 1) * st1 - self.weight_size + ks1
        output_pad1 = pad1 % 2
        pad1 = ((pad1 + output_pad1) / 2).astype(np.int)
        st1 = tuple(st1)
        pad1 = tuple(pad1)
        output_pad1 = tuple(output_pad1)

        self.branch3x3dbl_1 = BasicConvT2d(1, 1, kernel_size=1)
        self.branch3x3dbl_2 = BasicConvT2d(1, 1, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConvT2d(1, 1, kernel_size=3, stride=st3, padding=pad3, output_padding=output_pad3)

        self.branch3x3db2_1 = BasicConvT2d(1, 1, kernel_size=1)
        self.branch3x3db2_2 = BasicConvT2d(1, 1, kernel_size=3, stride=st3, padding=pad3, output_padding=output_pad3)

        self.branch1x1db1_1 = BasicConvT2d(1, 1, kernel_size=1, stride=st1, padding=pad1, output_padding=output_pad1)

        self.convfinal = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.insn = nn.InstanceNorm2d(1, eps=0.001)

    def forward(self, x):
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch3x3db2 = self.branch3x3db2_1(x)
        branch3x3db2 = self.branch3x3db2_2(branch3x3db2)

        branch1x1dbl = self.branch1x1db1_1(x)

        x = self.convfinal(torch.cat((branch3x3dbl,branch3x3db2,branch1x1dbl),1))
        return self.insn(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'weight_size=(' + str(self.weight_size[0]) + ',' \
            + str(self.weight_size[1]) + '),' \
            + 'param_size=(' + str(self.param_size[0]) + ',' \
            + str(self.param_size[1]) + '))'


class BasicConvT2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvT2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.insn = nn.InstanceNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.insn(x)
        return F.leaky_relu(x)


class Rule_list(nn.Module):
    def __init__(self, Rules, nums, param_size, weight_size):
        super(Rule_list, self).__init__()
        self.type_num = len(nums)
        self.structs = []
        self.nums = nums
        self.weight_size = weight_size
        for Rule, num in zip(Rules, nums):
            rule = Rule(param_size, weight_size)
            self.structs.append(rule.__repr__())
            name = rule.__class__.__name__
            self.add_module(name + '0', rule)
            for i in range(1, num):
                self.add_module(name + str(i), Rule(param_size,weight_size))

    def __call__(self, param_inner):
        return torch.cat([rule(param_inner).view(
            -1, 1, self.weight_size[0], self.weight_size[1]) for rule in self._modules.values()], 1)

    def __repr__(self):
        s = '\n'
        for i, struct, num in zip(range(self.type_num),self.structs, self.nums):
            s += '  (' + str(i) + '): ' + str(num) + ' * ' + struct + '\n'
        return s[:-1]


class _Part2WholeBase(ModuleParall):
    def __init__(self, config):
        super(_Part2WholeBase, self).__init__()
        self.param_channels = config['param_channels']
        self.param_sum_size = config['param_sum_size']
        self.hid_channels = config['hid_channels']
        self.hid_size = config['hid_size']
        self.len_weight = np.prod(config['weight_size'])
        self.weight_size = config['weight_size']
        self.lambd = config.get('lambd',0.01)
        if isinstance(self.weight_size, int):
            self.change_size2 = self.change_size1
        self.len_rules = config['len_rules']

        if self.param_sum_size % self.param_channels:
            raise ValueError('It must be a integer time between param_sum_size and param_channels!')
        self.param_size = self.param_sum_size // self.param_channels

        if self.hid_size % self.param_size != 0:
            raise ValueError('It must be a integer time change from param to hid!')
        self.time1 = self.hid_size // self.param_size

        if self.len_weight % self.hid_size != 0:
            raise ValueError('It must be a integer time change from hid to weight!')
        self.time2 = self.len_weight // self.hid_size

        self.ct1 = ChTimes(self.param_channels, self.hid_channels, self.time1, bias=False)\
            if not isinstance(self,Part2WholeWD) else \
            ChTimes(self.param_channels * 2, self.hid_channels, self.time1, bias=False)

    def change_size1(self,x):
        return x.reshape(-1, self.len_rules, self.weight_size)

    def change_size2(self,x):
        return x.reshape(-1, self.len_rules, self.weight_size[0], self.weight_size[1])


class Part2Whole(_Part2WholeBase):
    def __init__(self, config):
        super(Part2Whole, self).__init__(config)
        #self.bn1 = BatchWeight(self.hid_channels, config['num_iter'])
        self.bn1 = InstanceNormWeight(self.hid_channels, affine=True)
        self.ct2 = ChTimes(self.hid_channels, self.len_rules, self.time2)

    def forward(self, x, num):
        x = self.ct1(x)
        x_size = x.size()

        # TODO: Here 'x.view(-1, num...' has a mistake!
        x = F.elu(self.bn1(x.view(-1, num, x_size[1], x_size[2]).transpose(1,3).contiguous()
                           ).transpose(1,3).reshape(-1, x_size[1], x_size[2]))
        '''
        x = F.elu(self.bn1(x.view(num, -1, x_size[1], x_size[2]).transpose(1,3).contiguous()
                           ).transpose(1, 3).reshape(-1, x_size[1], x_size[2]))
        '''

        x = F.softshrink(self.change_size2(self.ct2(x).transpose(1,2)), self.lambd)
        return x


class Part2WholeBW(_Part2WholeBase):
    def __init__(self, config):
        super(Part2WholeBW, self).__init__(config)
        self.bn1 = BatchWeight(self.hid_channels, config['num_iter'])
        self.ct2 = ChTimes(self.hid_channels, self.len_rules, self.time2)

    def forward(self, x, num):
        x = self.ct1(x)
        x_size = x.size()
        x = F.elu(self.bn1(x.view(num, -1, x_size[1], x_size[2]).transpose(1,3).contiguous()
                           ).transpose(1, 3).reshape(-1, x_size[1], x_size[2]))
        x = F.softshrink(self.change_size2(self.ct2(x).transpose(1, 2)), self.lambd)
        return x


class Part2WholeR(Part2WholeBW):
    def __init__(self, config):
        super(Part2WholeR, self).__init__(config)

    def forward(self, x, num):
        x = self.ct1(x)
        x_size = x.size()
        x = F.elu(self.bn1(x.view(num, -1, x_size[1], x_size[2]).transpose(1,3).contiguous()
                           ).transpose(1, 3).reshape(-1, x_size[1], x_size[2]))
        x = self.change_size2(self.ct2(x).transpose(1, 2))
        return x


class Part2WholeWD(Part2Whole):
    def __init__(self, config):
        super(Part2WholeWD, self).__init__(config)

    def forward(self, x, num):
        num = x.size(0)
        x = torch.cat((x, x**2), dim=1)
        xs = []
        i = 0
        while i < num:
            x1 = F.elu(self.ln1(self.ct1(x[i:i+self.batch_size])))
            x1 = F.softshrink(self.change_size2(self.ct2(x1).transpose(1,2)), self.lambd)
            xs.append(x1)
            i += self.batch_size
        return torch.cat(xs,0)


class DropoutInner(nn.Dropout, ModuleParall):
    def __init__(self, p=0.5, inplace=False):
        super(DropoutInner, self).__init__(p, inplace)
    def inner_train(self, mode=True):
        self.inner_training = mode
        self.training = mode
        for module in self.children():
            if isinstance(module, ModuleParall):
                module.inner_train(mode)
        return self


class Part2WholeDP(Part2Whole):
    def __init__(self, config):
        super(Part2WholeDP, self).__init__(config)
        '''
        self.dp1 = MetaDropout(self.hid_channels * self.hid_size,
                               p=config['p'], p_last=config['p_last'],
                               t_max=config['t_max'], mode=config['mode'])
        self.dp2 = MetaDropout(self.len_rules * self.len_weight,
                               p=config['p'], p_last=config['p_last'],
                               t_max=config['t_max'], mode=config['mode'])
        '''
        self.dp1 = DropoutInner(p=config['p'])

    def forward(self, x, num):
        x = self.ct1(x)
        x_size = x.size()
        x = F.elu(self.bn1(x.view(-1, num, x_size[1], x_size[2]).transpose(1,3).contiguous()
                           ).transpose(1,3))
        #x = x.reshape(-1, num, x_size[1] * x_size[2])
        x = x.reshape(-1, x_size[1] * x_size[2])
        x = self.dp1(x).view(-1, x_size[1], x_size[2])
        x = self.ct2(x).transpose(1, 2)
        '''
        x_size2 = x.size()
        x = self.dp2(F.softshrink(x, self.lambd).reshape(-1, num, x_size2[1] * x_size2[2]))
        x = self.change_size2(x)
        '''
        x = self.change_size2(F.softshrink(x, self.lambd))
        return x

def main():
    pw = Part2WholeWD({'param_channels':6,'param_sum_size':6,
                  'hid_channels':8,'hid_size':9,
                  'len_rules':16,'weight_size':9,
                  'batch_size':20000})
    print(pw.ct1.weight.size())

class ChTimes(nn.Module):
    def __init__(self, in_channels, out_channels, times, bias=True):
        super(ChTimes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.times = times
        self.weight = Parameter(torch.Tensor(1, times, out_channels, in_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        batch_size = x.size(0)
        #print(x.size())
        x = x.view(batch_size, 1, -1, self.in_channels).matmul(self.weight.transpose(-2,-1))
        if self.bias is not None:
            x = x + self.bias
        return x.view(batch_size, -1, self.out_channels)

    def extra_repr(self):
        return 'times={times},out_channels={out_channels},in_channels={in_channels}'.format(**self.__dict__)


if __name__ == '__main__':
    main()