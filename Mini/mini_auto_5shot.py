#!/usr/bin/python3.6

import os

shot = 5
canny = False
model_name = 'leocgn'
#model_name = 'fuzzymeta1'
lossf = 'cross_entropy'

''''''
if os.system('python3 miniimagenet_e.py --train=True --picshow=False --metatrain_itr=30000'
             ' --meta_batch_size=8 --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --lossf='+lossf+' --canny='+str(canny)+' --model_name='+model_name):
    raise ValueError('Here1!')

# if os.system('python3 miniimagenet_e.py --resume=True --train=True --picshow=False --metatrain_itr=30000'
#              ' --resume_path=_i16400.pkl --meta_batch_size=8 --num_classes=5 --kshot='+str(shot)+''
#              ' --lossf='+lossf+' --kquery=15 --inner_num=3 --canny='+str(canny)'):
#     raise ValueError('Here1!')

if os.system('python3 miniimagenet_e.py --train=False --picshow=False --test_or_val=False --len_test=120'
             ' --test_interval=250 --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --lossf='+lossf+' --canny=' + str(canny)+' --model_name='+model_name):
    raise ValueError('Here2!')

if os.system('python3 miniimagenet_e.py --train=False --picshow=False --test_or_val=True'
             ' --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --lossf='+lossf+' --canny=' + str(canny)+' --model_name='+model_name):
    raise ValueError('Here3!')

if os.system('python3 miniimagenet_e.py --resume=True --train=True --picshow=False --metatrain_itr=60000'
             ' --resume_path=_i30000.pkl --meta_batch_size=8 --num_classes=5 --kshot='+str(shot)+''
             ' --lossf='+lossf+' --kquery=15 --inner_num=3 --canny='+str(canny)+' --model_name='+model_name):
    raise ValueError('Here4!')

if os.system('python3 miniimagenet_e.py --train=False --picshow=False --test_or_val=False --len_test=240'
             ' --test_interval=250 --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --lossf='+lossf+' --canny=' + str(canny)+' --model_name='+model_name):
    raise ValueError('Here5!')

if os.system('python3 miniimagenet_e.py --train=False --picshow=False --test_or_val=True'
             ' --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --lossf='+lossf+' --canny=' + str(canny)+' --model_name='+model_name):
    raise ValueError('Here6!')

if os.system('python3 miniimagenet_e.py --train=True --picshow=False --metatrain_itr=60000'
             ' --train_plus_val=True --meta_batch_size=8 --num_classes=5 --kshot='+str(shot)+''
             ' --lossf='+lossf+' --kquery=15 --inner_num=3 --canny='+str(canny)+' --model_name='+model_name):
    raise ValueError('Here7!')

if os.system('python3 miniimagenet_e.py --train=False --picshow=False --train_plus_val=True --test_or_val=True'
             ' --lossf='+lossf+' --num_classes=5 --kshot='+str(shot)+' --kquery=15 --inner_num=3'
             ' --canny='+str(canny)+' --model_name='+model_name):
    raise ValueError('Here8!')