#!/usr/bin/python3.6
""" Code for loading data. """

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

import numpy as np
import os
import random
from tqdm import tqdm
import pickle
from threading import Thread
import cv2

from utils import printl


class Sinusoid(object):
    def __init__(self, use_set='train', config={}):
        """
        Args:
            kshot: num samples to generate per class in one batch to train
            kquery: num samples to generate per class in one batch to test
            meta_batch_size: size of meta batch size (e.g. number of functions)
        """
        self.kshot = config.get('kshot')
        self.kquery = config.get('kquery')
        self.meta_batch_size = config.get('meta_batch_size')
        self.num_samples_per_class = self.kshot + self.kquery
        self.use_set = use_set

        self.generate = self.generate_sinusoid_batch
        self.amp_range = config.get('amp_range', [0.1, 5.0])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
        self.dim_input = 1
        self.dim_output = 1

    def __len__(self):
        if self.use_set == 'train':
            random.seed(1)
            np.random.seed(1)
            return 200000
        elif self.use_set == 'test':
            random.seed(2)
            np.random.seed(2)
            return 600
        elif self.use_set == 'val':
            random.seed(3)
            np.random.seed(3)
            return 600
        else:
            raise ValueError('Unrecognized data source')

    def __call__(self):
        for _ in range(len(self)):
            inputsa, outputsa, inputsb, outputsb, _, _ = self.generate()
            inputsa = torch.from_numpy(inputsa)
            outputsa = torch.from_numpy(outputsa)
            inputsb = torch.from_numpy(inputsb)
            outputsb = torch.from_numpy(outputsb)
            yield inputsa, outputsa, inputsb, outputsb

    def generate_sinusoid_batch(self):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.meta_batch_size]).astype(np.float32)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.meta_batch_size]).astype(np.float32)
        init_inputs = np.random.uniform(self.input_range[0], self.input_range[1],
                                              [self.meta_batch_size, self.num_samples_per_class, 1]).astype(np.float32)
        outputs = amp.reshape([self.meta_batch_size,1,1]) * np.sin(init_inputs - phase.reshape([self.meta_batch_size,1,1]))
        return init_inputs[:,:self.kshot], outputs[:,:self.kshot], \
               init_inputs[:,self.kshot:], outputs[:,self.kshot:], amp, phase


## Omni Image helper
def get_images(paths, num_classes, nb_samples=None, shuffle=True):
    labels = range(num_classes)
    rots = np.random.randint(num_classes,size=(num_classes))
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, np.rot90(Image.open(os.path.join(path, image)),rot)[np.newaxis,:]) \
        for i, rot, path in zip(labels, rots, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def get_files(folders, indexes, num_classes, nb_samples=None, shuffle=True):
    labels = range(num_classes)
    rots = np.random.randint(4,size=(num_classes))
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, rot, index * 20 + int(image[-6:-4]) - 1) \
        for i, rot, index in zip(labels, rots, indexes) \
        for image in sampler(os.listdir(folders[index]))]
    if shuffle:
        random.shuffle(images)
    return images


class Omniglot(Dataset):

    def __init__(self, use_set='train', resume_itr=0, config={}):
        """
        Args:
            kshot: num samples to generate per class in one batch to train
            kquery: num samples to generate per class in one batch to test
        """
        self.config = config
        self.use_set = use_set
        self.kshot = config.get('kshot')
        self.kquery = config.get('kquery')
        self.meta_batch_size = config.get('meta_batch_size', 1)
        self.resume_index = resume_itr * self.meta_batch_size
        self.resume_valindex = ((resume_itr - 1) // config.get('save_iter', 1000)
                                + 1) * self.meta_batch_size
        self.num_samples_per_class = self.kshot + self.kquery
        self.num_classes = config.get('num_classes', 5)
        self.num_shot_per_task = self.kshot * self.num_classes
        self.num_query_per_task = self.kquery * self.num_classes
        self.img_size = config.get('img_size', (28, 28))
        self.data_folder = config.get('data_folder', '../../../DataSets/omniglot')
        self.len_fname = len(self.data_folder) + 1
        self.tasks_file = 'omniglot_' + str(self.num_classes) + 'way_' + \
                          str(self.kshot) + 'shot_' + str(self.kquery) + 'query_'

        character_folders = [os.path.join(self.data_folder, family, character) \
                             for family in os.listdir(self.data_folder) \
                             if os.path.isdir(os.path.join(self.data_folder, family)) \
                             for character in os.listdir(os.path.join(self.data_folder, family))]
        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = config.get('num_train', 1200) - num_val
        self.train_folders = character_folders[:num_train]
        self.test_folders = character_folders[num_train + num_val:]
        self.val_folders = character_folders[num_train:num_train + num_val]

        if self.use_set == 'train':
            self.meta_character_folders = character_folders[:num_train]
            self.tasks_file = self.tasks_file + 'train.pkl'
            self.data_num = 400000
        elif self.use_set == 'test':
            self.meta_character_folders = character_folders[num_train + num_val:]
            self.tasks_file = self.tasks_file + 'test.pkl'
            self.data_num = 1800
        elif self.use_set == 'val':
            self.meta_character_folders = character_folders[num_train:num_train + num_val]
            self.tasks_file = self.tasks_file + 'val.pkl'
            self.data_num = 1800
        else:
            raise ValueError('Unrecognized data source')

    def change_use_set(self, use_set):
        if use_set == 'train':
            self.meta_character_folders = self.train_folders
            self.data_num = 400000
        elif use_set == 'test':
            self.meta_character_folders = self.test_folders
            self.data_num = 1800
        elif use_set == 'val':
            self.meta_character_folders = self.val_folders
            self.data_num = 1800
        else:
            raise ValueError('Unrecognized data source')
        self.tasks_file = self.tasks_file[:-len(self.use_set)-4] + use_set + '.pkl'
        self.use_set = use_set

    def make_tasks(self):
        random.seed(1)
        num_batch = self.data_num
        folders = self.meta_character_folders
        frange = range(len(folders))

        tasks = []
        for _ in tqdm(range(num_batch)):
            sampled_character_indexes = random.sample(frange, self.num_classes)
            random.shuffle(sampled_character_indexes)
            labels_and_images = get_files(folders, sampled_character_indexes, self.num_classes,
                                           nb_samples=self.num_samples_per_class, shuffle=False)
            labels, rots, images = [x for x in zip(*labels_and_images)]
            labels = np.array(labels,np.int8).reshape(self.num_classes,self.num_samples_per_class)
            labelsa = labels[:, :self.kshot].reshape(-1)
            labelsb = labels[:, self.kshot:].reshape(-1)
            rots = np.array(rots,np.int8).reshape(self.num_classes,self.num_samples_per_class)
            rotsa = rots[:, :self.kshot].reshape(-1)
            rotsb = rots[:, self.kshot:].reshape(-1)
            images = np.array(images,np.uint16).reshape(self.num_classes,self.num_samples_per_class)
            imagesa = images[:, :self.kshot].reshape(-1)
            imagesb = images[:, self.kshot:].reshape(-1)

            inda = np.arange(self.num_shot_per_task)
            np.random.shuffle(inda)
            indb = np.arange(self.num_query_per_task)
            np.random.shuffle(indb)
            tasks.append((labelsa[inda], rotsa[inda], imagesa[inda],
                          labelsb[indb], rotsb[indb], imagesb[indb]))

            #tasks.append((labelsa, rotsa, imagesa, labelsb, rotsb, imagesb))

        folders = tuple([x[self.len_fname:] + '/' + os.listdir(x)[0][-11:-6] for x in folders])
        tasks = tuple(tasks)
        with open(self.tasks_file, 'wb') as f:
            pickle.dump((tasks, folders), f)

    def __len__(self):
        if self.use_set == 'train':
            return self.data_num * self.meta_batch_size + 1
        elif self.config['train']:
            self.resume_index = self.resume_valindex
            return self.data_num * self.meta_batch_size + 1
        else:
            return self.data_num

    def make_one_task(self):
        folders = self.meta_character_folders
        sampled_character_folders = random.sample(folders, self.num_classes)
        random.shuffle(sampled_character_folders)
        labels_and_images = get_images(sampled_character_folders, self.num_classes,
                                       nb_samples=self.num_samples_per_class, shuffle=False)
        label, images = [x for x in zip(*labels_and_images)]
        label = np.array(label)
        images = 1. - (np.concatenate(images, axis=0) / 255.)
        inda = np.arange(self.num_shot_per_task)
        np.random.shuffle(inda)
        indb = np.arange(self.num_query_per_task)
        np.random.shuffle(indb)
        return torch.FloatTensor(images[:self.kshot][inda]), torch.LongTensor(label[:self.kshot][inda]),\
               torch.FloatTensor(images[self.kshot:][indb]), torch.LongTensor(label[self.kshot:][indb])

    def get_tasks(self):
        with open(self.tasks_file, 'rb') as f:
            self.tasks, self.folders = pickle.load(f)

        self.images = []
        index = []
        for i in range(1,21):
            if i < 10:
                index.append('0' + str(i) + '.jpg')
            else:
                index.append(str(i) + '.jpg')

        for folder in self.folders:
            for i in index:
                self.images.append(1 - (np.array(Image.open(os.path.join(self.data_folder,folder + i)))[np.newaxis,:] / 255.))
        self.images = np.concatenate(self.images,axis=0)

    def __getitem__(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labelsa = task[0].astype(np.int)
        rotsa = task[1].astype(np.int)
        indexa = task[2].astype(np.int)
        imagesa = np.zeros((self.num_shot_per_task, self.img_size[0], self.img_size[1]))
        for i in range(self.num_shot_per_task):
            imagesa[i] = np.rot90(self.images[indexa[i]], rotsa[i])
        imagesa = imagesa.reshape(self.num_shot_per_task, 1, self.img_size[0], self.img_size[1])

        labelsb = task[3].astype(np.int)
        rotsb = task[4].astype(np.int)
        indexb = task[5].astype(np.int)
        imagesb = np.zeros((self.num_query_per_task, self.img_size[0], self.img_size[1]))
        for i in range(self.num_query_per_task):
            imagesb[i] = np.rot90(self.images[indexb[i]], rotsb[i])
        imagesb = imagesb.reshape(self.num_query_per_task, 1, self.img_size[0], self.img_size[1])

        return (torch.FloatTensor(imagesa), torch.LongTensor(labelsa),
                torch.FloatTensor(imagesb), torch.LongTensor(labelsb))


class OmniglotHard(Omniglot):

    def __init__(self, use_set='train', resume_itr=0, config={}):
        super(OmniglotHard, self).__init__(use_set=use_set, resume_itr=resume_itr, config=config)

    def get_tasks(self):
        with open(self.tasks_file, 'rb') as f:
            self.tasks, self.folders = pickle.load(f)

    def __getitem__(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labelsa = task[0].astype(np.int)
        rotsa = task[1].astype(np.int)
        indexa = task[2].astype(np.int)
        imagesa = np.zeros((self.num_shot_per_task,self.img_size[0],self.img_size[1]))
        for i in range(self.num_shot_per_task):
            ind = indexa[i] // 20
            im_i = str(indexa[i] % 20 + 1)
            im_i = '0' + im_i + '.jpg' if len(im_i)==1 else im_i + '.jpg'
            imagesa[i] = np.rot90(Image.open(os.path.join(self.data_folder, self.folders[ind] + im_i)), rotsa[i])
        imagesa = 1 - imagesa.reshape(self.num_shot_per_task,1,self.img_size[0],self.img_size[1]) / 255.

        labelsb = task[3].astype(np.int)
        rotsb = task[4].astype(np.int)
        indexb = task[5].astype(np.int)
        imagesb = np.zeros((self.num_query_per_task,self.img_size[0],self.img_size[1]))
        for i in range(self.num_query_per_task):
            ind = indexb[i] // 20
            im_i = str(indexb[i] % 20 + 1)
            im_i = '0' + im_i + '.jpg' if len(im_i) == 1 else im_i + '.jpg'
            imagesb[i] = np.rot90(Image.open(os.path.join(self.data_folder, self.folders[ind] + im_i)), rotsb[i])
        imagesb = 1 - imagesb.reshape(self.num_query_per_task, 1, self.img_size[0], self.img_size[1]) / 255.

        return (torch.FloatTensor(imagesa), torch.LongTensor(labelsa),
                 torch.FloatTensor(imagesb), torch.LongTensor(labelsb))


## Mimi Image helper
def get_mimi_files(folders, indexes, num_classes, start=0,
                   each_class=600, nb_samples=None, shuffle=True):
    end = start + each_class
    labels = range(num_classes)
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, index * 600 + image) \
        for i, index in zip(labels, indexes) \
        for image in sampler(range(start, end))]
    if shuffle:
        random.shuffle(images)
    return images


class Lighting(object):
    def __init__(self, alphastd, eigval=None, eigvec=None):
        self.alphastd = alphastd
        self.eigval = eigval if eigval is not None else \
            torch.Tensor([[0.2175, 0.0188, 0.0045]]).repeat(3,1)
        self.eigvec = eigvec if eigvec is not None else \
            torch.Tensor([[ -0.5675,  0.7192,  0.4009 ],
                          [ -0.5808, -0.0045, -0.8140 ],
                          [ -0.5836, -0.6948,  0.4203 ]])
        self.eig = self.eigval * self.eigvec
        self.topilimage = T.ToPILImage()

    def __call__(self, x):
        if self.alphastd == 0:
            return x
        if x.size(0) == 1:
            x = x.repeat(3,1,1)

        alpha = torch.normal(mean=torch.zeros(3), std=self.alphastd
                             ).unsqueeze(0).repeat(3, 1)
        rgb = torch.sum(alpha * self.eig, dim=1).view(-1, 1, 1)
        return x + rgb

'''
if __name__ == '__main__':
    data_folder = '../../../DataSets/mini-imagenet'
    train_folders = [os.path.join(data_folder, 'train', family)
                     for family in os.listdir(os.path.join(data_folder, 'train'))
                     if os.path.isdir(os.path.join(data_folder, 'train', family))]

    test_folders = [os.path.join(data_folder, 'test', family)
                    for family in os.listdir(os.path.join(data_folder, 'test'))
                    if os.path.isdir(os.path.join(data_folder, 'test', family))]

    val_folders = [os.path.join(data_folder, 'val', family)
                   for family in os.listdir(os.path.join(data_folder, 'val'))
                   if os.path.isdir(os.path.join(data_folder, 'val', family))]

    for folder in train_folders:
        if len(os.listdir(folder)) != 600:
            print('Wrong!Wrong!Wrong!Wrong!Wrong!')

    for folder in test_folders:
        if len(os.listdir(folder)) != 600:
            print('Wrong!Wrong!Wrong!Wrong!Wrong!')

    for folder in val_folders:
        if len(os.listdir(folder)) != 600:
            print('Wrong!Wrong!Wrong!Wrong!Wrong!')

    print('End')
'''

class CatCanny(object):
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, img):
        img = self.transform1(img)
        canny = cv2.Canny(np.array(img), 200, 300) / 255.
        img2 = self.transform2(img)
        if img2.size(0) == 1:
            img2 = img2.repeat(3, 1, 1)
        img = torch.cat([img2, torch.from_numpy(canny).unsqueeze(0).
                        type(torch.FloatTensor)], dim=0)
        return img

'''
if __name__ == '__main__':
    img = Image.open("n0227997200000574.JPEG")

    img_size = 84
    normalize = T.Normalize(np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
    transform1 = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize(100),
        T.RandomResizedCrop(img_size),
    ])
    transform2 = T.Compose([
        T.ColorJitter(0.4, 0.4, 0.4),
        T.ToTensor(),
        Lighting(0.1),
        normalize,
    ])
    transform = CatCanny(transform1, transform2)
    print(transform(img))
'''

class MiniImagenet(Dataset):

    def __init__(self, use_set='train', resume_itr=0, config={}):
        """
        Args:
            kshot: num samples to generate per class in one batch to train
            kquery: num samples to generate per class in one batch to test
        """
        self.config = config
        self.use_set = use_set

        self.crosswise = config.get('crosswise')
        if self.crosswise:
            if self.use_set == 'train' :
                self.start = 0
                self.each_class = 480
                self.use_set += '_cross'
            elif self.use_set == 'val':
                self.start = 480
                self.each_class = 120
                self.use_set += '_cross'
        else:
            self.start = 0
            self.each_class = 600
        printl('use_set:' + self.use_set)

        self.kshot = config.get('kshot')
        self.kquery = config.get('kquery')
        self.meta_batch_size = config.get('meta_batch_size', 1)
        self.resume_index = resume_itr * self.meta_batch_size
        self.resume_valindex = ((resume_itr - 1) // config.get('save_iter', 1000)
                                + 1) * self.meta_batch_size
        self.num_samples_per_class = self.kshot + self.kquery
        self.num_classes = config.get('num_classes', 5)
        self.num_shot_per_task = self.kshot * self.num_classes
        self.num_query_per_task = self.kquery * self.num_classes
        self.img_size = config.get('img_size', 84)


        normalize = T.Normalize(np.array([0.485, 0.456, 0.406]),
                                np.array([0.229, 0.224, 0.225]))
        '''
        normalize = T.Normalize(np.array([0.5, 0.5, 0.5]),
                                np.array([0.5, 0.5, 0.5]))
        '''
        if config.get('train'):
            transform1 = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize(100),
                T.RandomResizedCrop(self.img_size),
            ])
            transform2 = T.Compose([
                T.ColorJitter(0.4, 0.4, 0.4),
                T.ToTensor(),
                Lighting(0.1),
                normalize,
            ])
        else:
            transform1 = T.Compose([
                T.Resize(100),
                T.CenterCrop(self.img_size),
            ])
            transform2 = T.Compose([
                T.ToTensor(),
                normalize,
            ])

        if config.get('canny', False):
            self.transform = CatCanny(transform1, transform2)
        else:
            self.transform = T.Compose([transform1, transform2])

        self.data_folder = config.get('data_folder', '../../../DataSets/mini-imagenet')
        self.encode_folder = config.get('data_folder', '../../../DataSets/mini-encode')
        self.len_fname = len(self.data_folder) + 1
        self.tasks_file = 'miniimagenet_' + str(self.num_classes) + 'way_' + \
                          str(self.kshot) + 'shot_' + str(self.kquery) + 'query_'

        self.train_folders = [os.path.join(self.data_folder, 'train', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'train')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'train', family))]

        self.test_folders = [os.path.join(self.data_folder, 'test', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'test')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'test', family))]

        self.val_folders = [os.path.join(self.data_folder, 'val', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'val')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'val', family))]

        if self.use_set == 'train_plus_val' or self.use_set == 'train_cross':
            self.meta_character_folders = self.train_folders + self.val_folders
            self.data_num = 400000
        elif self.use_set == 'val_cross':
            self.meta_character_folders = self.train_folders + self.val_folders
            self.data_num = 1000
        elif self.use_set == 'train':
            self.meta_character_folders = self.train_folders
            self.data_num = 400000
        elif self.use_set == 'test':
            self.meta_character_folders = self.test_folders
            self.data_num = 1000
        elif self.use_set == 'val':
            self.meta_character_folders = self.val_folders
            self.data_num = 1000
        else:
            raise ValueError('Unrecognized data source')
        self.tasks_file = self.tasks_file + self.use_set + '.pkl'

    def change_use_set(self, use_set):
        if use_set == 'train_plus_val' or use_set == 'train_cross':
            self.meta_character_folders = self.train_folders + self.val_folders
            self.data_num = 400000
        elif use_set == 'val_cross':
            self.meta_character_folders = self.train_folders + self.val_folders
            self.data_num = 1000
        elif use_set == 'train':
            self.meta_character_folders = self.train_folders
            self.data_num = 400000
        elif use_set == 'test':
            self.meta_character_folders = self.test_folders
            self.data_num = 1000
        elif use_set == 'val':
            self.meta_character_folders = self.val_folders
            self.data_num = 1000
        else:
            raise ValueError('Unrecognized data source')
        self.tasks_file = self.tasks_file[:-len(self.use_set)-4] + use_set + '.pkl'
        self.use_set = use_set
        if use_set == 'train_cross':
            self.start = 0
            self.each_class = 480
        elif use_set == 'val_cross':
            self.start = 480
            self.each_class = 120
        else:
            self.start = 0
            self.each_class = 600

    def make_tasks(self):
        random.seed(1)
        num_batch = self.data_num
        folders = self.meta_character_folders
        frange = range(len(folders))

        tasks = []
        for _ in tqdm(range(num_batch)):
            sampled_character_indexes = random.sample(frange, self.num_classes)
            random.shuffle(sampled_character_indexes)
            labels_and_images = get_mimi_files(folders, sampled_character_indexes, self.num_classes,
                                               start=self.start, each_class=self.each_class,
                                               nb_samples=self.num_samples_per_class, shuffle=False)
            labels, images = [x for x in zip(*labels_and_images)]
            labels = np.array(labels,np.int8).reshape(self.num_classes, self.num_samples_per_class)
            labelsa = labels[:, :self.kshot].reshape(-1)
            labelsb = labels[:, self.kshot:].reshape(-1)
            images = np.array(images,np.uint16).reshape(self.num_classes, self.num_samples_per_class)
            imagesa = images[:, :self.kshot].reshape(-1)
            imagesb = images[:, self.kshot:].reshape(-1)

            inda = np.arange(self.num_shot_per_task)
            np.random.shuffle(inda)
            indb = np.arange(self.num_query_per_task)
            np.random.shuffle(indb)
            tasks.append((labelsa[inda], imagesa[inda], labelsb[indb], imagesb[indb]))

            #tasks.append((labelsa, imagesa, labelsb, imagesb))

        folders = tuple([x[self.len_fname:] for x in folders])
        tasks = tuple(tasks)
        with open(self.tasks_file, 'wb') as f:
            pickle.dump((tasks, folders), f)

    def __len__(self):
        if self.use_set == 'train' or \
                        self.use_set == 'train_plus_val' or \
                        self.use_set == 'train_cross':
            return self.data_num * self.meta_batch_size + 1
        elif self.config['train']:
            self.resume_index = self.resume_valindex
            return self.data_num * self.meta_batch_size * 20 + 1

        else:
            return self.data_num

    def get_tasks(self):
        with open(self.tasks_file, 'rb') as f:
            self.tasks, self.folders = pickle.load(f)

        self.images = []
        for folder in self.folders:
            images_list = os.listdir(os.path.join(self.data_folder,folder))
            images_index = []
            for name in images_list:
                images_index.append(int(name[-13:-5]))
            images_index = np.array(images_index)
            images_list = np.array(images_list)
            images_list = images_list[images_index.argsort()]

            for name in images_list:
                self.images.append(Image.open(os.path.join(self.data_folder,folder,name)))

    def __getitem__(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labelsa = task[0].astype(np.int)
        indexa = task[1].astype(np.int)
        imagesa = []
        for i in range(self.num_shot_per_task):
            imagesa.append(self.transform(self.images[indexa[i]]).unsqueeze(0))
        imagesa = torch.cat(imagesa)

        labelsb = task[2].astype(np.int)
        indexb = task[3].astype(np.int)
        imagesb = []
        for i in range(self.num_query_per_task):
            imagesb.append(self.transform(self.images[indexb[i]]).unsqueeze(0))
        imagesb = torch.cat(imagesb)

        return (imagesa, torch.LongTensor(labelsa),
                imagesb, torch.LongTensor(labelsb))


class MiniImagenetHard(MiniImagenet):

    def __init__(self, use_set='train', resume_itr=0, config={}):
        super(MiniImagenetHard, self).__init__(use_set=use_set, resume_itr=resume_itr, config=config)
        if config['model_name'] == 'fuzzymeta_pre' or \
           config['model_name'] == 'fuzzycaps_pre':
            self.getitem = self.pre_getitem

    def get_tasks(self):
        with open(self.tasks_file, 'rb') as f:
            self.tasks, self.folders = pickle.load(f)

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labelsa = task[0].astype(np.int)
        indexa = task[1].astype(np.int)
        lsorta = np.argsort(labelsa)
        labelsa = labelsa[lsorta]
        indexa = indexa[lsorta]

        folders = []
        names = []
        for i in range(0, self.num_shot_per_task, self.kshot):
            ind = indexa[i] // 600
            images_list = os.listdir(os.path.join(self.data_folder, self.folders[ind]))
            images_index = []
            for name in images_list:
                images_index.append(int(name[-13:-5]))
            images_index = np.array(images_index)
            images_list = np.array(images_list)
            images_list = images_list[images_index.argsort()]
            folders.append(self.folders[ind])
            names.append(images_list)

        imagesa = []
        for i in range(self.num_shot_per_task):
            ind = i // self.kshot
            #print(os.path.join(self.data_folder, folders[ind], names[ind][indexa[i] % 600]))
            im = Image.open(os.path.join(self.data_folder, folders[ind], names[ind][indexa[i] % 600]))
            im = im.convert('RGB') if im.mode == 'L' else im
            im = self.transform(im)
            imagesa.append(im.unsqueeze(0))

        lsorta = np.argsort(lsorta)
        labelsa = labelsa[lsorta]
        imagesah = []
        for i in lsorta:
            imagesah.append(imagesa[i])
        imagesa = torch.cat(imagesah, dim=0)

        labelsb = task[2].astype(np.int)
        indexb = task[3].astype(np.int)
        lsortb = np.argsort(labelsb)
        labelsb = labelsb[lsortb]
        indexb = indexb[lsortb]

        imagesb = []
        for i in range(self.num_query_per_task):
            ind = i // self.kquery
            #print(os.path.join(self.data_folder, folders[ind], names[ind][indexb[i] % 600]))
            im = Image.open(os.path.join(self.data_folder, folders[ind], names[ind][indexb[i] % 600]))
            im = im.convert('RGB') if im.mode == 'L' else im
            im = self.transform(im)
            imagesb.append(im.unsqueeze(0))

        lsortb = np.argsort(lsortb)
        labelsb = labelsb[lsortb]
        imagesbh = []
        for i in lsortb:
            imagesbh.append(imagesb[i])
        imagesb = torch.cat(imagesbh, dim=0)

        return (imagesa, torch.LongTensor(labelsa),
                imagesb, torch.LongTensor(labelsb))

    def pre_getitem(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labelsa = task[0].astype(np.int)
        indexa = task[1].astype(np.int)

        folders = []
        names = []
        for i in range(0,self.num_shot_per_task, self.kshot):
            ind = indexa[i] // 600
            images_list = os.listdir(os.path.join(self.data_folder, self.folders[ind]))
            images_index = []
            for name in images_list:
                images_index.append(int(name[-13:-5]))
            images_index = np.array(images_index)
            images_list = np.array(images_list)
            images_list = images_list[images_index.argsort()]
            folders.append(self.folders[ind])
            names.append(images_list)

        imagesa = []
        for i in range(self.num_shot_per_task):
            ind = i // self.kshot
            im = np.load(os.path.join(self.encode_folder, folders[ind], names[ind][indexa[i] % 600][:-4] + 'npy'))
            im = torch.from_numpy(im).unsqueeze(0)
            imagesa.append(im)
        imagesa = torch.cat(imagesa, dim=0)

        labelsb = task[2].astype(np.int)
        indexb = task[3].astype(np.int)
        imagesb = []
        for i in range(self.num_query_per_task):
            ind = i // self.kquery
            im = np.load(os.path.join(self.encode_folder, folders[ind], names[ind][indexb[i] % 600][:-4] + 'npy'))
            im = torch.from_numpy(im).unsqueeze(0)
            imagesb.append(im)
        imagesb = torch.cat(imagesb, dim=0)

        return (imagesa, torch.LongTensor(labelsa),
                imagesb, torch.LongTensor(labelsb))

'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = {'kshot': 1, 'kquery': 15, 'meta_batch_size': 2,
              'data_folder': '../../DataSets/mini-imagenet',
              'train':True}
    dataset = MiniImagenetHard(use_set='train', resume_itr=0, config=config)
    dataset.get_tasks()
    print(config['meta_batch_size'])
    dataset = torch.utils.data.DataLoader(dataset, batch_size=config['meta_batch_size'],
                                               shuffle=False, num_workers=1)
    for data in dataset:
        imagesa, labelsa, imagesb, labelsb = data
        print('labelsa=', labelsa)
        print('imagesa=',imagesa)
        imagesa = ((imagesa + 1) / 2.).numpy().transpose(0, 1, 3, 4, 2)
        imagesb = ((imagesb + 1) / 2.).numpy().transpose(0, 1, 3, 4, 2)
        for i in range(config['meta_batch_size']):
            for j in range(5 * config['kshot']):
                plt.imshow(imagesa[i][j])
                plt.show()
            for j in range(5 * 5):
                plt.imshow(imagesb[i][j])
                plt.show()
        print('labelsb=', labelsb)
        break
'''

class MiniImagenetPre(Dataset):

    def __init__(self, use_set='train', config={}):
        """
        Args:
            kshot: num samples to generate per class in one batch to train
            kquery: num samples to generate per class in one batch to test
        """
        self.config = config
        self.use_set = use_set

        self.img_size = config.get('img_size', 84)
        self.data_folder = config.get('data_folder', '../../../DataSets/mini-imagenet')

        normalize = T.Normalize(np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
        if config['train']:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.Resize(100),
                T.RandomResizedCrop(self.img_size),
                T.ColorJitter(0.4, 0.4, 0.4),
                T.ToTensor(),
                Lighting(0.1),
                normalize,
            ])
        else:
            self.transform = T.Compose([
                T.Resize(100),
                T.CenterCrop(self.img_size),
                T.ToTensor(),
                normalize,
            ])

        self.train_folders = [os.path.join('train', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'train')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'train', family))]
        folders_index = [int(folder[-8:]) for folder in self.train_folders]
        folders_index = np.array(folders_index)
        self.train_folders = np.array(self.train_folders)
        self.train_folders = self.train_folders[folders_index.argsort()]
        self.train_folders = list(self.train_folders)

        self.test_folders = [os.path.join('test', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'test')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'test', family))]
        folders_index = [int(folder[-8:]) for folder in self.test_folders]
        folders_index = np.array(folders_index)
        self.test_folders = np.array(self.test_folders)
        self.test_folders = self.test_folders[folders_index.argsort()]
        self.test_folders = list(self.test_folders)

        self.val_folders = [os.path.join('val', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'val')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'val', family))]
        folders_index = [int(folder[-8:]) for folder in self.val_folders]
        folders_index = np.array(folders_index)
        self.val_folders = np.array(self.val_folders)
        self.val_folders = self.val_folders[folders_index.argsort()]
        self.val_folders = list(self.val_folders)

        self.meta_character_folders = self.train_folders + self.val_folders
        if config['train_plus_val']:
            self.start = 0
            self.end = 600
            printl('The train_plus_val is True!')
        elif self.use_set == 'train':
            self.start = 0
            self.end = 480
        elif self.use_set == 'val':
            self.start = 480
            self.end = 600
        else:
            raise ValueError('Unrecognized data source')
        self.num_each = self.end - self.start
        self.data_num = len(self.meta_character_folders) * self.num_each

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        label = index // self.num_each
        folder = os.path.join(self.data_folder, self.meta_character_folders[label])
        images_list = os.listdir(folder)
        images_index = []
        for name in images_list:
            images_index.append(int(name[-13:-5]))
        images_index = np.array(images_index)
        images_list = np.array(images_list)
        images_list = images_list[images_index.argsort()]
        images_list = images_list[self.start:self.end]

        image = self.transform(Image.open(
            os.path.join(folder, images_list[index % self.num_each])))
        if image.size(0) == 1:
            image = image.repeat(3,1,1)
        return image, label


class MiniImagenetEncode(Dataset):

    def __init__(self, config={}):
        """
        Args:
            kshot: num samples to generate per class in one batch to train
            kquery: num samples to generate per class in one batch to test
        """
        self.config = config

        self.batch_size = config.get('batch_size', 256)
        self.img_size = config.get('img_size', 84)
        self.data_folder = config.get('data_folder', '../../../DataSets/mini-imagenet')
        self.encode_folder = config.get('data_folder', '../../../DataSets/mini-encode')
        if not os.path.exists(self.encode_folder):
            os.mkdir(self.encode_folder)
        if not os.path.exists(os.path.join(self.encode_folder, 'train')):
            os.mkdir(os.path.join(self.encode_folder, 'train'))
        if not os.path.exists(os.path.join(self.encode_folder, 'test')):
            os.mkdir(os.path.join(self.encode_folder, 'test'))
        if not os.path.exists(os.path.join(self.encode_folder, 'val')):
            os.mkdir(os.path.join(self.encode_folder, 'val'))

        self.transform = T.Compose([
            T.Resize(100),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(np.array([0.485, 0.456, 0.406]),
                        np.array([0.229, 0.224, 0.225])),
        ])

        self.train_folders = [os.path.join('train', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'train')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'train', family))]
        folders_index = [int(folder[-8:]) for folder in self.train_folders]
        folders_index = np.array(folders_index)
        self.train_folders = np.array(self.train_folders)
        self.train_folders = self.train_folders[folders_index.argsort()]
        self.train_folders = list(self.train_folders)

        self.test_folders = [os.path.join('test', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'test')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'test', family))]
        folders_index = [int(folder[-8:]) for folder in self.test_folders]
        folders_index = np.array(folders_index)
        self.test_folders = np.array(self.test_folders)
        self.test_folders = self.test_folders[folders_index.argsort()]
        self.test_folders = list(self.test_folders)

        self.val_folders = [os.path.join('val', family) \
                             for family in os.listdir(os.path.join(self.data_folder,'val')) \
                             if os.path.isdir(os.path.join(self.data_folder, 'val', family))]
        folders_index = [int(folder[-8:]) for folder in self.val_folders]
        folders_index = np.array(folders_index)
        self.val_folders = np.array(self.val_folders)
        self.val_folders = self.val_folders[folders_index.argsort()]
        self.val_folders = list(self.val_folders)

        self.meta_character_folders = self.train_folders + self.val_folders + self.test_folders
        for folder in self.meta_character_folders:
            if not os.path.exists(os.path.join(self.encode_folder, folder)):
                os.mkdir(os.path.join(self.encode_folder, folder))
        self.num_each = 600
        self.data_num = len(self.meta_character_folders) * self.num_each

    def __len__(self):
        return self.data_num

    def __call__(self, model):
        def load_image(i, image_h):
            image = self.transform(Image.open(
                os.path.join(self.data_folder, image_h)))
            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)
            image = image.unsqueeze(0)
            ims_feat[i] = image

        threads = []

        images_queue = []
        for folder in tqdm(self.meta_character_folders):
            images_list = os.listdir(os.path.join(self.data_folder, folder))
            images_index = []
            for name in images_list:
                images_index.append(int(name[-13:-5]))
            images_index = np.array(images_index)
            images_list = np.array(images_list)
            images_list = list(images_list[images_index.argsort()])
            images_list = [os.path.join(folder, image) for image in images_list]
            images_queue.extend(images_list)
            while len(images_queue) >= self.batch_size:
                images_h = images_queue[:self.batch_size]
                images_queue = images_queue[self.batch_size:]

                ims_feat = [None] * self.batch_size
                for i, image_h in enumerate(images_h):
                    t = Thread(target=load_image, args=(i, image_h))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                ims_feat = torch.cat(ims_feat, dim=0).cuda()
                ims_encode = model(ims_feat).cpu().numpy()
                for im, image in zip(ims_encode, images_h):
                    t = Thread(target=np.save, args=(
                        os.path.join(self.encode_folder, image[:-5]), im))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

        images_h = images_queue

        ims_feat = [None] * len(images_h)
        for i, image_h in enumerate(images_h):
            t = Thread(target=load_image, args=(i, image_h))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        ims_feat = torch.cat(ims_feat, dim=0).cuda()
        ims_encode = model(ims_feat).cpu().numpy()
        for im, image in zip(ims_encode, images_h):
            t = Thread(target=np.save, args=(
                os.path.join(self.encode_folder, image[:-5]), im))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()