'''
@author: georgeretsi
'''

import os
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset

from os.path import isfile

from auxilary_functions import image_resize, centered

# IAM CONFIGURATION !!
trainset_file_ext = 'set_split/trainset.txt'
testset_file_ext = 'set_split/testset.txt'
valset_file_ext = 'set_split/validationset1.txt'

line_file_ext = 'ascii/lines.txt'
word_file_ext = 'ascii/words.txt'

word_path_ext = 'words'
line_path_ext = 'lines'

stopwords_path = 'iam-stopwords'

saved_datasets_path = './saved_datasets'

def gather_iam_info(path, set='train', level='word'):

    # train/test file
    if set == 'train':
        valid_set = np.loadtxt(os.path.join(path, trainset_file_ext), dtype=str)
    elif set == 'test':
        valid_set = np.loadtxt(os.path.join(path, testset_file_ext), dtype=str)
    elif set == 'val':
        valid_set = np.loadtxt(os.path.join(path, valset_file_ext), dtype=str)
    else:
        print('shitloader')
        return


    if level == 'word':
        gtfile= os.path.join(path, word_file_ext)
        root_path = os.path.join(path, word_path_ext)
    elif level == 'line':
        gtfile = os.path.join(path, line_file_ext)
        root_path = os.path.join(path, line_path_ext)
    else:
        print('shitloader')
        return

    gt = []
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]

            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            if level == 'word':
                line_name = pathlist[-2]
                del pathlist[-2]
            elif level == 'line':
                line_name = pathlist[-1]

            if (info[1] != 'ok') or (line_name not in valid_set):
                continue

            img_path = '/'.join(pathlist)

            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))

    return gt

def main_loader(path, set, level):

    info = gather_iam_info(path, set, level)

    data = []
    for i, (img_path, transcr) in enumerate(info):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

        try:
            img = img_io.imread(img_path + '.png')
            img = 1 - img.astype(np.float32) / 255.0
            img = image_resize(img, height=img.shape[0] // 2)
        except:
            continue

        data += [(img, transcr.replace("|", " "))]

    return data

class IAMLoader(Dataset):

    def __init__(self, path, tset, level, fixed_size=(128, None), transforms=None):

        self.transforms = transforms
        self.set = tset
        self.fixed_size = fixed_size

        save_file = saved_datasets_path + '/' + tset + '_' + level + '.pt'

        if isfile(save_file) is False:

            if not os.path.isdir(saved_datasets_path):
                os.mkdir(saved_datasets_path)

            data = main_loader(path, set=tset, level=level)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)

        self.data = data
        self.character_classes = list(sorted(set(list(''.join([t for _,t in data])))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]

        transcr = " " + self.data[index][1] + " "

        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        if self.set == 'train':
            # random resize at training !!!
            nwidth = int(np.random.uniform(.5, 1.5) * img.shape[1])
            nheight = int((np.random.uniform(.8, 1.2) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        # small pads!!
        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

        # pad with zeroes
        img = centered(img, (fheight, fwidth), border_value=0.0)

        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)

        img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr
