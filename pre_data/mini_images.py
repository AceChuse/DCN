"""
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)
Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the directory and put '../../DataSets/mini-imagenet/'.
Put the the images in the directory '../../DataSets/mini-imagenet/new_dataset/'.
Then run this script from the miniImagenet directory:
    cd pre_data/
    python3 mini_images.py
"""

from __future__ import print_function
import csv
import glob
import os

from PIL import Image

path_to_images = '../../DataSets/mini-imagenet/'

all_images = glob.glob(path_to_images + 'new_dataset/*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((100, 100), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)

# Put in correct directory
for datatype in ['train', 'val', 'test']:
    os.system('mkdir ' + path_to_images + datatype)

    with open(path_to_images + datatype + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_label = ''
        for i, row in enumerate(reader):
            label = row[1]
            image_name = row[0]
            if label != last_label:
                cur_dir = path_to_images + datatype + '/' + label + '/'
                os.system('mkdir ' + cur_dir)
                last_label = label
            os.system('mv ' + path_to_images + 'new_dataset/' + image_name + ' ' + cur_dir)