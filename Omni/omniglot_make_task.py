#!/usr/bin/python3.6

import sys
import os
filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
sys.path.append(filepath)
from DataSolver import *

# 5-way 1-shot 1-query
print('1111')
random.seed(1)
np.random.seed(1)
dataset = Omniglot(use_set='train', config={'num_classes':5,'kshot':1,'kquery':1})
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()

print('2222')
# 5-way 5-shot 5-query
random.seed(1)
np.random.seed(1)
dataset = Omniglot(use_set='train', config={'num_classes':5,'kshot':5,'kquery':5})
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()

print('3333')
# 20-way 1-shot 1-query
random.seed(1)
np.random.seed(1)
dataset = Omniglot(use_set='train', config={'num_classes':20,'kshot':1,'kquery':1})
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()

print('4444')
# 20-way 5-shot 5-query
random.seed(1)
np.random.seed(1)
dataset = Omniglot(use_set='train', config={'num_classes':20,'kshot':5,'kquery':5})
dataset.make_tasks()

random.seed(2)
np.random.seed(2)
dataset.change_use_set('val')
dataset.make_tasks()

random.seed(3)
np.random.seed(3)
dataset.change_use_set('test')
dataset.make_tasks()