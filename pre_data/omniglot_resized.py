"""
Usage instructions:
    First download the omniglot dataset
    and put the contents of both images_background and images_evaluation
    in ../../DataSets/omniglot/ (without the root folder)
    Then, run the following:
    cd pre_data/
    python omniglot_resize.py
"""
from PIL import Image
import glob
import os

image_path = '../../DataSets/omniglot/*/*/'

all_images = glob.glob(image_path + '*')

i = 0

for image_file in all_images:
    im = Image.open(image_file)
    im = im.resize((28,28), resample=Image.LANCZOS)
    os.remove(image_file)
    im.save(image_file[:-4]+'.jpg')
    i += 1

    if i % 200 == 0:
        print(i)