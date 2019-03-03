#!/usr/bin/python3.6

"""
Usage Instructions:
    10-shot sinusoid:

        python3 sinusoid_e.py --kshot=10 --kquery=10 --inner_num=2 --train=True --metatrain_itr=60000 --model_name=fuzzymeta
        python3 sinusoid_e.py --kshot=10 --kquery=10 --inner_num=20 --train=False --test_or_val=True --model_name=fuzzymeta

    To run evaluation, use the '--train=False' flag, and the '--test_or_val=False' flag to use the validation set
    '--test_or_val=True' flag to use the test set.

    Choose the test iteration by '--test_iter=-1', when test on the validation set it would test last '--len_test' models,
    when test on the test set it would test just one model.
"""

import argparse
import re
import sys
import os

import torch.optim as optim

filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
sys.path.append(filepath)
from DataSolver import *
from fuzzymeta import *
from utils import *


def get_sinu_fmeta(config, path=None):
    set_num = 2
    hidden_size = 40
    inner_num = config['inner_num'] if config['train'] else config['inner_num_test']
    inner_num_train = config['inner_num']

    class Part2WholeSine(object):
        def __new__(cls, len_rules, param_sum_size, weight_size, inner_num, param_num):
            config = {'param_channels': 10, 'param_sum_size': param_sum_size,
                      'hid_channels': 8, 'hid_size': 40,
                      'len_rules': len_rules, 'weight_size': weight_size,
                      'num_iter': inner_num_train}
            return Part2WholeBW(config)

    fnet = FNetParall(param_size=100,
                      weight_size=(hidden_size, hidden_size),
                      Rule=Part2WholeSine,
                      len_rules=2 ** set_num,
                      set_num=set_num,
                      inner_num=inner_num,
                      inner_lr=config['inner_lr'],
                      match=CapChMatch(hidden_size, set_num, 1, retain_grad=True),
                      layers=(nn.Linear(1, hidden_size),
                              nn.ReLU(),
                              View(hidden_size,1,1),
                              FCh2D(hidden_size, hidden_size, 1, bias=False),
                              nn.ReLU(),
                              FCh2D(hidden_size, hidden_size-5, 1, bias=False),
                              View(hidden_size-5),
                              nn.ReLU(),
                              nn.Linear(hidden_size-5, 1)
                              ))

    amsgrad = True
    if not config['resume'] and config['train']:
        print_network(fnet, 'fnet')
        printl('meta_batch_size=' + str(config['meta_batch_size']))
    optimf = optim.Adam(fnet.parameters(), lr=config['meta_lr'], betas=(0.9, 0.999), weight_decay=0.,amsgrad=amsgrad)

    if path != None:
        if os.path.isfile(path):
            print("=> loading fnet '{}'".format(path))
            module = torch.load(path)
            fnet.load_state_dict(module)

    return MetaNet(fnet, F.mse_loss, optimf, save_path=config['save_path'], parall_num=config['meta_batch_size'],
                   save_iter=config['save_iter'], print_iter=1000)


def get_sinu_maml(config, path=None):
    hidden_size = 40
    inner_num = config['inner_num'] if config['train'] else config['inner_num_test']

    mnet = MAML(inner_num=inner_num,
                inner_lr=config['inner_lr'],
                layers=(nn.Linear(1, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size, bias=False),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size-5, bias=False),
                        nn.ReLU(),
                        nn.Linear(hidden_size-5, 1)
                        ))

    amsgrad = True
    if not config['resume'] and config['train']:
        print_network(mnet, 'mnet')
        printl('meta_batch_size=' + str(config['meta_batch_size']))
    optimf = optim.Adam(mnet.parameters(), lr=config['meta_lr'], betas=(0.9, 0.999), weight_decay=0., amsgrad=amsgrad)

    if path != None:
        if os.path.isfile(path):
            print("=> loading fnet '{}'".format(path))
            module = torch.load(path)
            mnet.load_state_dict(module)

    return MetaNet(mnet, F.mse_loss, optimf, save_path=config['save_path'], parall_num=config['meta_batch_size'],
                   save_iter=config['save_iter'], print_iter=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is pytorch network.")
    parser.add_argument("--baseline", "-bl", help="oracle or None", default=None)
    parser.add_argument("--metatrain_itr", "-mi", help="number of metatraining iterations.", default=60000, type=int) # 15k for omniglot, 50k for sinusoid
    parser.add_argument("--meta_batch_size", "-mbs", help="number of tasks sampled per meta-update", default=25, type=int)
    parser.add_argument("--meta_lr", "-ml", help="the base learning rate of the generator", default=0.001, type=float)
    parser.add_argument("--kshot", "-ks", help="number of examples used for inner gradient update (K for K-shot learning).", default=20, type=int)
    parser.add_argument("--kquery", "-kq", help="number of examples used for inner test (K for K-query).", default=20, type=int)
    parser.add_argument("--inner_lr", "-ilr", help="step size alpha for inner gradient update.", default=1e-3, type=float)
    parser.add_argument("--inner_num", "-inu", help="number of inner gradient updates during training.", default=2, type=int)
    parser.add_argument("--inner_num_test", "-int", help="number of inner gradient updates during test.", default=30,type=int)

    ## Logging, saving, and testing options
    parser.add_argument("--logdir", "-ld", help="directory for summaries and checkpoints.",
                        default='../../Result/sinu')
    parser.add_argument("--resume", "-rs", help="resume training if there is a model available", default=False, type=ast.literal_eval)
    parser.add_argument("--train", "-tr", help="True to train, False to test.", default=True, type=ast.literal_eval)
    parser.add_argument("--test_iter", "-ti", help="iteration to load model (-1 for latest model)", default=-1, type=int)
    parser.add_argument("--test_or_val", "-tov",
                        help="Set to true to test on the the test set, False for the validation set.", default=False, type=ast.literal_eval)
    parser.add_argument("--len_test", "-lt",
                        help="The number of models using in test.", default=5, type=int)
    parser.add_argument("--bestn", help="The best n model.", default=1, type=int)

    parser.add_argument("--save_iter", "-si", help="iteration to save model", default=1000, type=int)
    parser.add_argument("--resume_path", "-lp", help="path to load model and file.",
                        default='_i4000.pkl')
    parser.add_argument("--model_name", "-mn", help="The name of model used in the task.",
                        default='fuzzymeta')#'maml')
    config = parser.parse_args().__dict__

    tasks_file = str(config['kshot']) + 'shot_' + str(config['kquery']) + 'query'

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

    if not config['resume'] and config['train']:
        date = re.sub('[: ]', '-', str(time.asctime(time.localtime(time.time()))))
    else:
        with open(os.path.join(config['save_path'], 'model_training.txt'), "r") as f:
            date = f.read().strip('\n')

    config['logdir'] = os.path.join(config['save_path'], date + '.txt')
    if config['model_name'] == 'fuzzymeta':
        get_module = get_sinu_fmeta
    elif config['model_name'] == 'maml':
        get_module = get_sinu_maml

    set_resultpath(config['logdir'])
    if config['train']:

        resume_itr = int(re.findall('_i([0-9]*)\.pkl$', config['resume_path'])[0]) - 1 \
            if config['resume'] else 0
        config['resume_path'] = config['resume_path'] if config['resume'] else None

        trainset = Sinusoid('train',config)
        valset = Sinusoid('val', config)

        torch.manual_seed(1)
        metanet = get_module(config, config['resume_path'])
        metanet.train()
        metanet.fit(trainset(), valset(), config['metatrain_itr'], resume_itr)

        valset = Sinusoid('val', config)
        metanet.eval()
        means, stds, ci95 = metanet.test(valset())

        printl('Mean validation accuracy/loss, stddev, and confidence intervals')
        printl((means, stds, ci95))

        plt_result(config['save_path'], iter=1000)
    else:
        inner_num = config['inner_num'] if config['train'] else config['inner_num_test']
        printl('inner_num=', str(inner_num) + '\n')

        on_set = 'test' if config['test_or_val'] else 'val'
        test_iter = np.load(config['save_path'] + '/last_iter.npy') \
            if config['test_iter'] == -1 else config['test_iter'] * config['save_iter']

        def test():
            global on_set
            dataset = Sinusoid('test', config) if config['test_or_val'] else Sinusoid('val',config)

            model = '_i' + str(config['test_iter']) + '.pkl'
            load_path = config['save_path'] + '/' + model
            metanet = get_module(config, load_path)
            metanet.eval()
            mean, std, ci95 = metanet.test(dataset(), classify=False)
            printl(model + ' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (mean, std, ci95))
            return mean, std, ci95

        if not config['test_or_val']:
            means = []
            stds = []
            ci95s = []
            test_iters = [test_iter + i * config['save_iter'] for i in range(-config['len_test'] + 1, 1)]

            for test_iter in test_iters:
                config['test_iter'] = test_iter
                mean, std, ci95 = test()
                printl("")
                means.append(mean)
                stds.append(std)
                ci95s.append(ci95)

            best = np.argmin(means)
            model = '_i' + str(test_iters[best]) + '.pkl'
            printl('Iter%d is the best accub_val: ' % best)
            printl(model + ' loss on ' + on_set + ': means=%.6f, stds=%.6f, ci95=%.6f' % (
            means[best], stds[best], ci95s[best]))
        else:
            config['test_iter'] = test_iter
            test()
            printl('')