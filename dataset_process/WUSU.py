import os
import numpy as np
from torch.utils import data
import utils.transform as transform
import imageio

size = 256
num_classes = 13

root = '../WUSU/'

ST_COLORMAP = [[255, 255, 255], [220, 220, 220], [0, 0, 0], [255, 211, 127], [255, 0, 0], [255, 235, 175], [178, 178, 178], 
               [38, 155, 0], [177, 255, 0], [0, 197, 255], [0, 92, 230], [0, 255, 197], [197, 0, 255], [178, 178, 178]]

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def normalize_img(img):
    """Normalize image by subtracting mean and dividing by std."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

    return normalized_img


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class Data(data.Dataset):
    def __init__(self, data_list, type='train', data_loader=img_loader):
        self.dataset_path = os.path.join(root, type)
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

    def __transforms(self, aug, t1_img, t2_img, t1_label, t2_label):
        if aug:
            t1_img, t2_img, t1_label, t2_label = transform.rand_rot90_flip_MCD(t1_img, t2_img, t1_label, t2_label)

        t1_img = normalize_img(t1_img)  # imagenet normalization
        t1_img = np.transpose(t1_img, (2, 0, 1))

        t2_img = normalize_img(t2_img)  # imagenet normalization
        t2_img = np.transpose(t2_img, (2, 0, 1))

        return t1_img, t2_img, t1_label, t2_label

    def __getitem__(self, index):
        t1_path = os.path.join(self.dataset_path, 'Rgb_15', self.data_list[index])
        t2_path = os.path.join(self.dataset_path, 'Rgb_18', self.data_list[index])
        T1_label_path = os.path.join(self.dataset_path, 'mask_15', self.data_list[index])
        T2_label_path = os.path.join(self.dataset_path, 'mask_18', self.data_list[index])


        t1_img = self.loader(t1_path)
        t2_img = self.loader(t2_path)

        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)

        if 'train' in self.data_pro_type:
            t1_img, t2_img, t1_label, t2_label = self.__transforms(True, t1_img, t2_img, t1_label, t2_label)
        else:
            t1_img, t2_img, t1_label, t2_label = self.__transforms(False, t1_img, t2_img, t1_label, t2_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        return t1_img, t2_img, t1_label, t2_label

    def __len__(self):
        return len(self.data_list)
