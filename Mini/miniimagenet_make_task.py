#!/usr/bin/python3.6

import sys
import os
filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
sys.path.append(filepath)
from DataSolver import *

# 5-way 1-shot 1-query
print('1111')
random.seed(0)
np.random.seed(0)
dataset = MiniImagenet(use_set='train_plus_val', config={'num_classes':5,'kshot':1,'kquery':15})
dataset.make_tasks()

random.seed(1)
np.random.seed(1)
dataset.change_use_set('train')
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()

random.seed(4)
np.random.seed(4)
dataset.change_use_set('train_cross')
dataset.make_tasks()

random.seed(5)
np.random.seed(5)
dataset.change_use_set('val_cross')
dataset.make_tasks()

print('2222')
# 5-way 5-shot 5-query
random.seed(1)
np.random.seed(1)
dataset = MiniImagenet(use_set='train_plus_val', config={'num_classes':5,'kshot':5,'kquery':15})
dataset.make_tasks()

random.seed(1)
np.random.seed(1)
dataset.change_use_set('train')
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()

random.seed(4)
np.random.seed(4)
dataset.change_use_set('train_cross')
dataset.make_tasks()

random.seed(5)
np.random.seed(5)
dataset.change_use_set('val_cross')
dataset.make_tasks()