#!/usr/bin/python3.6

import argparse
import re
import sys
import os

import torch.optim as optim
import torch.utils.data as torchdata
from torch.optim.lr_scheduler import MultiStepLR

filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
sys.path.append(filepath)
from DataSolver import *
from fuzzymeta import *
from utils import *


class Part2WChOmni(object):
    def __new__(cls, len_rules, param_sum_size, weight_size, inner_num, param_num):
        config = {'param_channels': 32, 'param_sum_size': param_sum_size,
                  'hid_channels': 16, 'hid_size': 64,
                  'len_rules': len_rules, 'weight_size': weight_size,
                  'num_iter': inner_num}
        return Part2WholeBW(config)


def get_omni_fmeta(config, path=None):
    set_num = 4
    channels = 64
    kernel_size = 3

    if config['num_classes'] == 20:
        if config['kshot'] == 1:
            parall_num = config['meta_batch_size'] // 1
        elif config['kshot'] == 5:
            parall_num = config['meta_batch_size'] // 2 \
                if config['model_name'] == 'fuzzymeta6' else 6
    elif config['num_classes'] == 5:
        if config['kshot'] == 1:
            parall_num = config['meta_batch_size'] // 1
        elif config['kshot'] == 5:
            parall_num = config['meta_batch_size'] // 2

    if config['meta_batch_size'] % parall_num == 1:
        raise ValueError("Can't residual paral num 1!")


    if config['model_name'] == 'fuzzymeta4':
        fnet = FWeDParall(param_size=32 * 32,
                          weight_size=(channels, channels),
                          Rule=Part2WChOmni,
                          len_rules=2 ** set_num,
                          set_num=set_num,
                          inner_num=config['inner_num'],
                          inner_lr=config['inner_lr'],
                          inner_wd=config['inner_wd'],
                          match=CapChMatch(channels, set_num, kernel_size, retain_grad=True),
                          layers=(NonFineTune(nn.Conv2d(1, channels, kernel_size, stride=1, padding=1, bias=False),
                                              nn.BatchNorm2d(channels),
                                              nn.ReLU(inplace=True),
                                              nn.AvgPool2d(2),
                                              nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1, bias=False),
                                              nn.BatchNorm2d(channels),
                                              nn.ReLU(inplace=True),
                                              nn.AvgPool2d(2),
                                              ),
                                  FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                  nn.ReLU(),
                                  FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                  nn.ReLU(),
                                  nn.AvgPool2d(5),
                                  View(channels),
                                  nn.Linear(channels, config['num_classes'])
                                  ))
    elif config['model_name'] == 'fuzzymeta6':
        fnet = FWeDParall(param_size=32 * 32,
                          weight_size=(channels, channels),
                           Rule=Part2WChOmni,
                           len_rules=2 ** set_num,
                           set_num=set_num,
                           inner_num=config['inner_num'],
                           inner_lr=config['inner_lr'],
                           inner_wd=config['inner_wd'],
                           match=CapChMatch(channels, set_num, kernel_size, retain_grad=True),
                           layers=(NonFineTune(nn.Conv2d(1, channels, kernel_size, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(channels),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1,
                                                         bias=False),
                                               nn.BatchNorm2d(channels),
                                               nn.ReLU(inplace=True),
                                               nn.AvgPool2d(2),
                                               nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1,
                                                         bias=False),
                                               nn.BatchNorm2d(channels),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(channels, channels, kernel_size, stride=1, padding=0,
                                                         bias=False),
                                               nn.BatchNorm2d(channels),
                                               nn.ReLU(inplace=True),
                                               nn.AvgPool2d(2),
                                               ),
                                   FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                   nn.ReLU(),
                                   FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                   nn.ReLU(),
                                   nn.AvgPool2d(5),
                                   View(channels),
                                   nn.Linear(channels, config['num_classes'])
                                   ))

    amsgrad = True
    if not config['resume'] and config['train']:
        print_network(fnet, 'fnet')
        printl('meta_batch_size=' + str(config['meta_batch_size']))
        printl('lossf=' + config['lossf'])
        printl('inner_super=' + str(config["inner_super"]))
        printl('parall_num=' + str(parall_num))

    if isinstance(fnet, FLeRParall):
        lr_id = [id(fnet.lr_pw), id(fnet.lr_pfl)]
        lr_params = [fnet.lr_pw, fnet.lr_pfl]
    elif isinstance(fnet, FWeDParall):
        lr_id = [id(fnet.lr_pw), id(fnet.lr_pfl), id(fnet.lr_pwed), id(fnet.lr_pflwed)]
        lr_params = [fnet.lr_pw, fnet.lr_pfl, fnet.lr_pwed, fnet.lr_pflwed]

    if type(fnet) == FNetParall or type(fnet) == MAML:
        params = fnet.parameters()
    else:
        base_params = filter(lambda p: id(p) not in lr_id, fnet.parameters())
        params = [{'params': lr_params, 'lr': config['meta_lr'] * config['inner_super'],
                   'initial_lr': config['meta_lr'] * config['inner_super']},
                  {'params': base_params, 'lr': config['meta_lr'], 'initial_lr': config['meta_lr']}]

    optimf = optim.Adam(params, lr=config['meta_lr'], betas=(0.9, 0.999), weight_decay=0., amsgrad=amsgrad)
    resume_itr = config.get('resume_itr', 0)
    if config['num_classes'] == 20:
        if config['kshot'] == 1:
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
        elif config['kshot'] == 5:
            config['parall_num'] = 6#config['meta_batch_size'] // 2
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
    elif config['num_classes'] == 5:
        if config['kshot'] == 1:
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
        elif config['kshot'] == 5:
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)

    if not config['resume'] and config['train']:
        printl(optimf)
        printl(sche2str(scheduler))
    optimf = Scheduler(scheduler, optimf)

    if path != None:
        if os.path.isfile(path):
            print("=> loading fnet '{}'".format(path))
            module = torch.load(path)
            fnet.load_state_dict(module)

    if config['lossf'] == 'margin_loss':
        lossf = margin_loss
    elif config['lossf'] == 'cross_entropy':
        lossf = F.cross_entropy
    return MetaNet(fnet, lossf, optimf, save_path=config['save_path'], parall_num=parall_num,
                   save_iter=config['save_iter'], print_iter=100)


def get_omni_fcaps(config, path=None):
    set_num = 4
    channels = 64
    kernel_size = 3
    if config['num_classes'] == 20:
        if config['kshot'] == 1:
            parall_num = config['meta_batch_size'] // 2
        elif config['kshot'] == 5:
            parall_num = config['meta_batch_size'] // 8
    elif config['num_classes'] == 5:
        if config['kshot'] == 1:
            parall_num = config['meta_batch_size'] // 1
        elif config['kshot'] == 5:
            parall_num = config['meta_batch_size'] // 2

    if config['meta_batch_size'] % parall_num == 1:
        raise ValueError("Can't residual paral num 1!")

    if config['model_name'] == 'fuzzycaps4':
        fnet = FWeDParall(param_size=32 * 32,
                          weight_size=(channels, channels),
                          Rule=Part2WChOmni,
                          len_rules=2 ** set_num,
                          set_num=set_num,
                          inner_num=config['inner_num'],
                          inner_lr=config['inner_lr'],
                          inner_wd=config['inner_wd'],
                          match=CapChMatch(channels, set_num, kernel_size, retain_grad=True),
                          layers=(NonFineTune(nn.Conv2d(1, channels, kernel_size, stride=1, padding=1, bias=False),
                                              nn.BatchNorm2d(channels),
                                              nn.ReLU(inplace=True),
                                              nn.AvgPool2d(2),
                                              nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1,
                                                        bias=False),
                                              nn.BatchNorm2d(channels),
                                              nn.ReLU(inplace=True),
                                              nn.AvgPool2d(2),
                                              ),
                                  FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                  nn.ReLU(),
                                  FCh2D(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(channels, affine=True, track_running_stats=False),
                                  nn.ReLU(),
                                  View(channels, -1),
                                  CapsuleShare(channels, config['num_classes'] * 5, routings=3, bias=True, retain_grad=True),
                                  View(config['num_classes'], 5),
                                  Norm()
                                  ))

    amsgrad = True
    if not config['resume'] and config['train']:
        print_network(fnet, 'fnet')
        printl('meta_batch_size=' + str(config['meta_batch_size']))
        printl('lossf=' + config['lossf'])
        printl('inner_super=' + str(config["inner_super"]))
        printl('parall_num=' + str(parall_num))

    if isinstance(fnet, FLeRParall):
        lr_id = [id(fnet.lr_pw), id(fnet.lr_pfl)]
        lr_params = [fnet.lr_pw, fnet.lr_pfl]
    elif isinstance(fnet, FWeDParall):
        lr_id = [id(fnet.lr_pw),id(fnet.lr_pfl),id(fnet.lr_pwed),id(fnet.lr_pflwed)]
        lr_params = [fnet.lr_pw, fnet.lr_pfl, fnet.lr_pwed, fnet.lr_pflwed]

    if type(fnet) == FNetParall:
        params = fnet.parameters()
    else:
        base_params = filter(lambda p: id(p) not in lr_id, fnet.parameters())
        params = [{'params': lr_params, 'lr': config['meta_lr'] * config['inner_super'],
                   'initial_lr': config['meta_lr'] * config['inner_super']},
                  {'params': base_params, 'lr': config['meta_lr'], 'initial_lr': config['meta_lr']}]

    optimf = optim.Adam(params, lr=config['meta_lr'], betas=(0.9, 0.999), weight_decay=0., amsgrad=amsgrad)
    resume_itr = config.get('resume_itr', 0)
    if config['num_classes'] == 20:
        if config['kshot'] == 1:
            config['parall_num'] = config['meta_batch_size'] // 2
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
        elif config['kshot'] == 5:
            sys.setrecursionlimit(100000)
            config['parall_num'] = config['meta_batch_size'] // 8
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
    elif config['num_classes'] == 5:
        if config['kshot'] == 1:
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)
        elif config['kshot'] == 5:
            scheduler = LambdaLR(optimf, MSCCALR(milestones=[10000, 20000, 30000, 40000, 50000, 60000],
                                                 gamma=0.5, interval=2000), last_epoch=resume_itr - 1)


    if not config['resume'] and config['train']:
        printl(optimf)
        printl(sche2str(scheduler))
    optimf = Scheduler(scheduler, optimf)

    if path != None:
        if os.path.isfile(path):
            print("=> loading fnet '{}'".format(path))
            module = torch.load(path)
            fnet.load_state_dict(module)

    if config['lossf'] == 'margin_loss':
        lossf = margin_loss
    elif config['lossf'] == 'cross_entropy':
        lossf = F.cross_entropy
    return MetaNet(fnet, lossf, optimf, save_path=config['save_path'], parall_num=parall_num,
                   save_iter=config['save_iter'], print_iter=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is pytorch network.")
    parser.add_argument("--num_classes", "-nc",
                        help="number of classes used in classification (e.g. 5-way classification).", default=5, type=int)
    parser.add_argument("--pretrain_itr", "-pi", help="number of pre-training iterations.", default=0, type=int)
    parser.add_argument("--metatrain_itr", "-mi", help="number of metatraining iterations.", default=60000,
                        type=int)  # 15k for omniglot, 50k for sinusoid
    parser.add_argument("--meta_batch_size", "-mbs", help="number of tasks sampled per meta-update", default=32,
                        type=int)
    parser.add_argument("--meta_lr", "-ml", help="the base learning rate of the model", default=0.001, type=float)
    parser.add_argument("--lossf", "-lf", help="the loss function of the model", default='cross_entropy', type=str)
    parser.add_argument("--kshot", "-ks",
                        help="number of examples used for inner gradient update (K for K-shot learning).", default=1,
                        type=int)
    parser.add_argument("--kquery", "-kq", help="number of examples used for inner test (K for K-query).", default=1,
                        type=int)
    parser.add_argument("--inner_wd", "-iwd", help="weight decay for inner gradient update.", default=1.,
                        type=float)
    parser.add_argument("--inner_lr", "-ilr", help="step size alpha for inner gradient update.", default=0.4,
                        type=float)
    parser.add_argument("--inner_num", "-inu", help="number of inner gradient updates during training.", default=1,
                        type=int)
    parser.add_argument("--inner_super", "-ins", help="step size alpha for inner lr and weight decay.", default=0.1,
                        type=float)

    ## Logging, saving, and testing options
    parser.add_argument("--logdir", "-ld", help="directory for summaries and checkpoints.",
                        default='../../Result/omni')
    parser.add_argument("--picshow", "-pw", help="show picture of curve.", default=True, type=ast.literal_eval)
    parser.add_argument("--resume", "-rs", help="resume training if there is a model available", default=False,
                        type=ast.literal_eval)
    parser.add_argument("--train", "-tr", help="True to train, False to test.", default=True, type=ast.literal_eval)
    parser.add_argument("--train_plus_val", default=False, type=ast.literal_eval)
    parser.add_argument("--test_iter", "-ti", help="iteration to load model (-1 for latest model)", default=-1,
                        type=int)
    parser.add_argument("--test_or_val", "-tov",
                        help="Set to true to test on the the test set, False for the validation set.", default=False,
                        type=ast.literal_eval)
    parser.add_argument("--ensembles_val", "-ebv", default=True, type=ast.literal_eval)
    parser.add_argument("--ensembles", "-eb", default=True, type=ast.literal_eval)
    parser.add_argument("--len_test", "-lt",
                        help="The number of models using in test.", default=10, type=int)

    parser.add_argument("--save_iter", "-si", help="iteration to save model", default=1000, type=int)
    parser.add_argument("--test_interval", "-tint", help="interval to test model", default=1000, type=int)
    parser.add_argument("--resume_path", "-lp", help="path to load model and file.",
                        default='_i57000.pkl')
    parser.add_argument("--model_name", "-mn", help="The name of model used in the task.",
                        default='fuzzymeta4')
    config = parser.parse_args().__dict__

    tasks_file = str(config['num_classes']) + 'way_' + \
                          str(config['kshot']) + 'shot_' + str(config['kquery']) + 'query'

    parent_direct = os.path.split(config['logdir'])[0]
    if not os.path.exists(parent_direct):
        os.mkdir(parent_direct)

    if not os.path.exists(config['logdir']):
        os.mkdir(config['logdir'])
    config['logdir'] = os.path.join(config['logdir'], config['model_name'])
    if not os.path.exists(config['logdir']):
        os.mkdir(config['logdir'])
    config['resume_path'] = os.path.join(config['logdir'], tasks_file, config['resume_path'])
    config['save_path'] = os.path.join(config['logdir'], tasks_file)
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])

    if not config['resume'] and config['train'] and not config['train_plus_val']:
        date = re.sub('[: ]', '-', str(time.asctime(time.localtime(time.time()))))
    else:
        with open(os.path.join(config['save_path'], 'model_training.txt'), "r") as f:
            date = f.read().strip('\n')

    config['logdir'] = os.path.join(config['save_path'], date + '.txt')
    if config['model_name'] == 'fuzzymeta4' or \
        config['model_name'] == 'fuzzymeta6':
        get_module = get_omni_fmeta
    elif config['model_name'] == 'fuzzycaps4':
        get_module = get_omni_fcaps

    set_resultpath(config['logdir'])
    if config['train']:

        resume_itr = int(re.findall('_i([0-9]*)\.pkl$', config['resume_path'])[0]) \
            if config['resume'] else 0
        config['resume_itr'] = resume_itr
        config['resume_path'] = config['resume_path'] if config['resume'] else None

        use_set = 'train_plus_val' if config['train_plus_val'] else 'train'
        trainset = Omniglot(use_set, resume_itr, config)
        trainset.get_tasks()
        trainset = torch.utils.data.DataLoader(trainset, batch_size=config['meta_batch_size'],
                                    shuffle=False, num_workers=8)
        valset = Omniglot('val', resume_itr, config)
        valset.get_tasks()
        valset = torch.utils.data.DataLoader(valset, batch_size=config['meta_batch_size'],
                                               shuffle=False, num_workers=8)

        torch.manual_seed(1)
        metanet = get_module(config, config['resume_path'])
        metanet.train()
        metanet.fit(trainset, valset, config['metatrain_itr'], resume_itr)

        config['train'] = False
        valset = Omniglot('val', config=config)
        valset.get_tasks()
        valset = torch.utils.data.DataLoader(valset, batch_size=config['meta_batch_size'],
                                             shuffle=False, num_workers=8)
        metanet.eval()
        means, stds, ci95, means_accu, stds_accu, ci95_accu = metanet.test(valset, classify=True)

        '''
        with open(config['logdir'], "w") as log:
            log.write("")
        '''
        printl('Loss on val: means=%.6f, stds=%.6f, ci95=%.6f' % (means, stds, ci95))
        printl('Accu on val: means=%.6f, stds=%.6f, ci95=%.6f' % (means_accu, stds_accu, ci95_accu))

        plt_result(config['save_path'], iter=config['save_iter'], picshow=config['picshow'])
    else:
        if config['test_interval'] % config['save_iter'] != 0:
            raise ValueError('The test_interval should be an integer multiple of save_iter!')
        printl('inner_num=', str(config['inner_num']) + '\n')
        testset = Omniglot('test', config=config) if config['test_or_val'] else Omniglot('val', config=config)
        testset.get_tasks()

        if not config['test_or_val']:
            if config['ensembles_val']:
                ensembles_val(get_module, testset, config, load_y=False, pattern='No_down')
            else:
                module_val(get_module, testset, config)
        else:
            if config['ensembles']:
                ensembles_test(get_module, testset, 'test', config, load_y=False, pattern='No_down')
            else:
                config['test_iter'] = np.load(config['save_path'] + '/last_iter.npy') \
                    if config['test_iter'] == -1 else config['test_iter'] * config['test_interval']
                module_test(get_module, testset, 'test', config)