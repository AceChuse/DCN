#!/usr/bin/python3.6

import argparse
import sys
import os
import ast
import re
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models.resnet import BasicBlock
import math

filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
sys.path.append(filepath)
from DataSolver import *
from utils import *
from densenet import *


def to_var(data):
    if use_cuda:
        return Variable(data[0]).cuda(), Variable(data[1]).cuda()
    else:
        return Variable(data[0]), Variable(data[1])


class ResNet_Mini(nn.Module):

    def __init__(self, block, layers, num_classes=80):
        super(ResNet_Mini, self).__init__()
        self.inplanes = 64
        inplanes = self.inplanes
        self.block = block
        self.num_classes = num_classes
        self.layers = layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(6, stride=1)
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_embedding(self):
        return nn.Sequential(self.conv1, self.bn1, self.relu, self.layer1,
                             self.layer2,self.layer3,self.layer4)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + self.block.__name__ + ',' \
            + 'layers=' + str(self.layers) + ',' \
            + 'num_classes=' + str(self.num_classes) + ')'


def resnet34_mini(path=None, requires_grad=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Mini(Block2d313, [3, 4, 6, 3], **kwargs)
    if path is not None:
        model.load_state_dict(torch.load(path))
        print("=> loading fnet '{}'".format(path))
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad_(False)
    else:
        print_network(model, 'resnet34')
    return model


class Dense_Mini(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=80):
        super(Dense_Mini, self).__init__()
        self.hidden_size = 256
        self.features = nn.Sequential(nn.Conv2d(3, num_init_features, kernel_size=5, stride=2, padding=0, bias=False),
                                      nn.BatchNorm2d(num_init_features),
                                      nn.ReLU(inplace=True),)

        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        dense = DenseEncode(growth_rate=growth_rate, block_config=block_config,
                            num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate)
        self.num_last_features = dense.num_features

        self.features.add_module('relu-2', nn.ReLU())
        self.features.add_module('dense', dense)
        self.features.add_module('conv-2', nn.Conv2d(self.num_last_features, self.hidden_size, kernel_size=1,
                                                       stride=1, padding=0, bias=False))
        self.features.add_module('norm-1', nn.BatchNorm2d(self.hidden_size))
        self.features.add_module('relu-1', nn.ReLU())
        self.features.add_module('avg_pool', nn.AvgPool2d(5))
        self.features.add_module('view', ViewNonFT(-1))

        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

    def get_embedding(self):
        return self.features

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'growth_rate=' + str(self.growth_rate) + ',' \
               + 'block_config=' + str(self.block_config) + ',\n' \
               + 'num_init_features=' + str(self.num_init_features) + ',\n' \
               + 'bn_size=' + str(self.bn_size) + ',' \
               + 'drop_rate=' + str(self.drop_rate) + ',\n' \
               + 'num_last_features=' + str(self.num_last_features) + ',\n' \
               + 'num_classes=' + str(self.num_classes) + ')'


def dense_mini(path=None, requires_grad=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Dense_Mini(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                       **kwargs)
    if path is not None:
        model.load_state_dict(torch.load(path))
        print("=> loading fnet '{}'".format(path))
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad_(False)
    else:
        print_network(model, 'dense')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is pre-trained feature.")

    # Folder options
    parser.add_argument("--logdir", "-ld", help="directory for summaries and checkpoints.",
                        default='../../../Result/mini_pre')
    parser.add_argument('--model', default='dense', type=str)
    parser.add_argument('--train_plus_val', default=False, type=ast.literal_eval)

    # Training options
    parser.add_argument("--train", "-tr", help="True to train, False to test.",
                        default=False, type=ast.literal_eval)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epoch_step', default='[80,100,120,140,160,180]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.8, type=float)
    parser.add_argument('--resume', default=0, type=int)

    # Testing options
    parser.add_argument("--test_epoch", "-te", help="epoch to load model (-1 for latest model)",
                        default=298, type=int)
    parser.add_argument("--test_intervel", "-ti", default=2, type=int)
    parser.add_argument("--test_len", "-lt", help="The number of models using in test.",
                        default=10, type=int)

    # Encode options
    parser.add_argument("--encode", "-e", default=True, type=ast.literal_eval)

    config = parser.parse_args().__dict__
    epoch_step = json.loads(config['epoch_step'])
    num_classes = 80

    config['save_path'] = os.path.join(config['logdir'], config['model'])
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])

    if config['resume'] == 0 and config['train']:
        date = re.sub('[: ]', '-', str(time.asctime(time.localtime(time.time()))))
    else:
        with open(os.path.join(config['save_path'], 'model_training.txt'), "r") as f:
            date = f.read().strip('\n')
    config['logdir'] = os.path.join(config['save_path'], date + '.txt')
    set_resultpath(config['logdir'])

    if config['resume'] == 0:
        model_path = None
    else:
        model_path = os.path.join(config['save_path'], '_i' + str(config['resume']) + '.pkl')

    if config['model'] == 'resnet34':
        get_model = resnet34_mini
    elif config['model'] == 'dense':
        get_model = dense_mini

    if config['train']:
        model = get_model(num_classes=num_classes, path=model_path)
        model.cuda()
        model.train(config['train'])

        optim = optim.SGD([{'params':model.parameters(), 'lr': config['lr'],
                            'initial_lr': config['lr']}], momentum=0.9,
                          weight_decay=config['weight_decay'], nesterov=True)
        '''
        scheduler = MultiStepLR(optim, milestones=epoch_step,
                                gamma=config['lr_decay_ratio'], last_epoch=config['resume'])
        '''
        scheduler = CosineAnnealingLR(optim, config['epochs'], 1e-5)
        dataset = MiniImagenetPre(use_set='train', config=config)

        '''
        import matplotlib.pyplot as plt
        plt.imshow(dataset[0][0].numpy().transpose(1,2,0) / 2. + 0.5)
        plt.show()
        print(dataset[0][0])
        '''
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                                 shuffle=True, num_workers=8)

        for epoch in range(config['resume'] + 1, config['epochs'] + 1):
            scheduler.step()
            loss_all = 0.
            accu_all = 0.
            data_num = 0
            for data in dataloader:
                feat, label = to_var(data)
                optim.zero_grad()
                output = model(feat)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optim.step()

                accu_all += torch.sum((torch.argmax(output, 1) == label).type(FloatTensor))

                batch_size = len(label)
                loss_all += batch_size * float(loss)
                data_num += batch_size
            printl('[%d]: loss=%.4f, accu=%.4f' % (epoch, loss_all / data_num, accu_all / data_num))
            torch.save(model.state_dict(),
                       os.path.join(config['save_path'], '_i' + str(epoch) + '.pkl'))
            np.save(os.path.join(config['save_path'],'last_iter'), np.array(epoch, dtype=np.int))
            with open(os.path.join(config['save_path'], 'model_training.txt'), "w") as f:
                f.write(date)
    elif not config['encode']:
        dataset = MiniImagenetPre(use_set='val', config=config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                                 shuffle=True, num_workers=6)

        def test():
            model_name = '_i' + str(test_epoch) + '.pkl'
            model_path = os.path.join(config['save_path'], model_name)
            model = get_model(num_classes=num_classes, requires_grad=config['train'], path=model_path)
            model.cuda()
            model.train(config['train'])

            loss_all = 0.
            accu_all = 0.
            data_num = 0
            for data in dataloader:
                feat, label = to_var(data)
                output = model(feat).detach()
                loss = F.cross_entropy(output, label)

                accu_all += torch.sum((torch.argmax(output, 1) == label).type(FloatTensor))

                batch_size = len(label)
                loss_all += batch_size * float(loss)
                data_num += batch_size
            printl((model_name + ' on ' + dataset.use_set +': loss=%.4f, accu=%.4f'
                    ) % (loss_all / data_num, accu_all / data_num))

            del model
            gc.collect()

        test_epoch = np.load(os.path.join(config['save_path'], 'last_iter.npy')) \
            if config['test_epoch'] == -1 else config['test_epoch']
        test_epochs = [test_epoch + i * config['test_intervel']
                       for i in range(-config['test_len'] + 1, 1)]
        for test_epoch in test_epochs:
            test()
    else:
        test_epoch = np.load(os.path.join(config['save_path'], 'last_iter.npy')) \
            if config['test_epoch'] == -1 else config['test_epoch']
        model_path = os.path.join(config['save_path'], '_i' + str(test_epoch) + '.pkl')
        model = get_model(num_classes=num_classes, requires_grad=config['train'], path=model_path)
        dataset = MiniImagenetEncode(config=config)
        dataset(model.get_embedding().cuda())