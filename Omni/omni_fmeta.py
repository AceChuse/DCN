#!/usr/bin/python3.6

import os

shot = 1
nc = 5
model_name = 'fuzzymeta4'
lossf = 'cross_entropy'
inner_num = 1

query = shot
ms = 32 if nc == 5 else 16
''''''
if os.system('python3 omniglot_e.py --train=True --picshow=False --metatrain_itr=60000'
             ' --meta_batch_size='+str(ms)+' --num_classes='+str(nc)+' --kshot='+str(shot)+''
             ' --lossf='+lossf+' --kquery='+str(query)+' --inner_num='+str(inner_num)+' --model_name='+model_name):
    raise ValueError('Here1!')

# if os.system('python3 omniglot_e.py --resume=True --train=True --picshow=False --metatrain_itr=60000'
#              ' --resume_path=_i4300.pkl --meta_batch_size='+str(ms)+' --num_classes='+str(nc)+''
#              ' --lossf='+lossf+' --kshot='+str(shot)+' --kquery='+str(query)+' --inner_num='+str(inner_num)+' --model_name='+model_name):
#     raise ValueError('Here1!')

if os.system('python3 omniglot_e.py --train=False --picshow=False --test_or_val=False --len_test=60'
             ' --lossf='+lossf+' --test_interval=1000 --num_classes='+str(nc)+' --kshot='+str(shot)+''
             ' --kquery='+str(query)+' --inner_num='+str(inner_num)+' --model_name='+model_name):
    raise ValueError('Here2!')

if os.system('python3 omniglot_e.py --train=False --picshow=False --test_or_val=True'
             ' --num_classes='+str(nc)+' --kshot='+str(shot)+' --kquery='+str(query)+' --inner_num='+str(inner_num)+''
             ' --lossf='+lossf+' --model_name='+model_name):
    raise ValueError('Here3!')