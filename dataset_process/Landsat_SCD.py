import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
from torchvision.transforms import functional as F

size = 416
num_classes = 5
ClASSES = ['0: No change', '1: Farmland', '2: Desert', '3: Building', '4: Water']
COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]

MEAN_A = np.array([141.53, 139.20, 137.73] )
STD_A = np.array([81.99, 83.31, 83.89])
MEAN_B = np.array([137.36, 136.50, 135.144])
STD_B = np.array([85.97, 86.01, 86.81])


root = '../Landsat-SCD_dataset/'


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im


def tensor2int(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = im * STD_A + MEAN_A
    else:
        im = im * STD_B + MEAN_B
    return im.astype(np.uint8)


def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def read_RSimages(mode, rescale=False):
    assert mode in ['train', 'val', 'test']
    list_path=os.path.join(root, mode+'_list.txt')
    img_A_dir = os.path.join(root, 'A')
    img_B_dir = os.path.join(root, 'B')
    label_A_dir = os.path.join(root, 'labelA')
    label_B_dir = os.path.join(root, 'labelB')
    
    list_info = open(list_path, 'r')
    data_list = list_info.readlines()
    data_list = [item.rstrip() for item in data_list]
    
    imgsA_list, imgsB_list, labelsA, labelsB = [], [], [], []
    count = 0
    for it in data_list:
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            imgsA_list.append(img_A_path)
            imgsB_list.append(img_B_path)
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            labelsA.append(label_A)
            labelsB.append(label_B)
        count+=1
        if not count%100: print('%d/%d images loaded.'%(count, len(data_list)))
        #if count>99: break
    print(str(len(imgsA_list)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgsA_list, imgsB_list, labelsA, labelsB


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)

    def __len__(self):
        return len(self.imgs_list_A)


class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
                
        imgA_dir = os.path.join(test_dir, 'A')
        imgB_dir = os.path.join(test_dir, 'B')
        list_path=os.path.join(test_dir, 'test_list.txt')
        list_info = open(list_path, 'r')
        data_list = list_info.readlines()
        data_list = [item.rstrip() for item in data_list]
        
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len
