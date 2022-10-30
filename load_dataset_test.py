#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import glob
import torch
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import torch.optim as optim


import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

import torch.nn.functional as F

from models import build_model
import utils


def get_train_files(input_folder, n_frames, step=None, prefix='Train'):

    folder_level1 = sorted(os.listdir(input_folder))
    output = []
    for folder_name in folder_level1:
        if prefix in folder_name:
            filenames = sorted(glob.glob("%s/%s/*.tif" % (input_folder, folder_name)))
            basename = [os.path.basename(f) for f in filenames]

            labels = [0 for _ in range(len(filenames))]
            
            start = 0
            while start + n_frames < len(filenames):
                list_files = filenames[start:start + n_frames]
                output += [{'list_files': list_files,
                           'start_frame_idx': start,
                           'end_frame_idx': start+ n_frames - 1,
                           'prefix': prefix,
                           'label': 0,
                           'video_name': folder_name
                           }]
                start += step if step is not None else n_frames
            
            
    return output

def get_gt_label(filenames, threshold = 0.2):
    cnt = 0
    
    for i, fname in enumerate(filenames):
        img = np.float32(cv2.imread(fname, 0))/255.
        if np.sum(img) > 0.001:
            cnt += 1
            
    return int(cnt >= 0.2*len(filenames))


def get_test_files(input_folder, n_frames, step=None, prefix='Test', train_split=[1,2,3,4,5]):

    folder_level1 = sorted(os.listdir(input_folder))
    output_trains, output_tests = [], []
    
    for folder_name in folder_level1:
        if (prefix in folder_name) and ('gt' not in folder_name):
            test_id = int(folder_name[-3:])
            print(test_id)
            filenames = sorted(glob.glob("%s/%s/*.tif" % (input_folder, folder_name)))
            gt_filenames = sorted(glob.glob("%s/%s_gt/*.bmp" % (input_folder, folder_name)))
            basename = [os.path.basename(f) for f in filenames]

            labels = [0 for _ in range(len(filenames))]
            
            start = 0
            while start + n_frames < len(filenames):
                list_files = filenames[start:start + n_frames]
                data = {'list_files': list_files,
                           'start_frame_idx': start,
                           'end_frame_idx': start+ n_frames - 1,
                           'prefix': prefix,
                           'label': get_gt_label(gt_filenames[start:start + n_frames]),
                           'video_name': folder_name
                           }
                start += step if step is not None else n_frames
                
                if test_id in train_split:
                    output_trains += [data]
                else:
                    output_tests += [data]
            
            
    return output_trains, output_tests


def load_images_and_label(filenames, image_size, local_transform, normalize_transform):
    """Loads an image into a tensor. Also returns its label."""
    imgs = []

    for i, fname in enumerate(filenames):
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #if local_transform is not None:
            #res = local_transform(image=img)
            #img = res['image']

        img = cv2.resize(img, (image_size, image_size))
        img = torch.tensor(img).permute((2, 0, 1)).float().div(255)
        img = normalize_transform(img)
        imgs.append(img)

    return torch.stack(imgs, dim=0).reshape((-1, image_size, image_size))


class VideoDataset(Dataset):
    def __init__(self, output_files, n_frames, image_size, normalize_transform,
                 device,
                 split,
                 seed=10):

        self.image_size = image_size
        self.device = device
        self.split = split

        self.video_imgs = output_files

        self.list_labels = [a['label'] for a in self.video_imgs]

        self.transform = None
        self.normalize_transform = normalize_transform
        if self.split == 'train':
            self.transform = albumentations.Compose([
                            Rotate((-15., 15.), p=0.5),
                            ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1),
                            HorizontalFlip(p=0.2),
                            RandomBrightnessContrast(p=0.5, brightness_limit=0.5, contrast_limit=0.5),
                            GaussNoise(p=0.2),
                            albumentations.OneOf([
                            RandomGamma(gamma_limit=(60, 120), p=1.),
                            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=20, p=1.)
                            ], p=0.2)
                            ])

    def __getitem__(self, index):
        filenames = self.video_imgs[index]['list_files']
        label = self.list_labels[index]
        imgs = load_images_and_label(filenames,
                                     self.image_size,
                                     self.transform,
                                     self.normalize_transform)

        return imgs, label

    def __len__(self):
        return len(self.video_imgs)

    def get_labels(self):
        return self.list_labels



# Always get features prediction
def get_predictions(model, data_loader, device):
    predicts = []
    features = []
    
    model = model.eval()
    for batch_idx, data in enumerate(data_loader):
        batch_size = data[0].shape[0]
        
        x = data[0].to(device)
        y_true = data[1].to(device).long()

        y_pred, fts = model(x, features=True)

        predicts += y_pred.detach().cpu().numpy().tolist()
        features += fts.detach().cpu().numpy().tolist()
        
    return np.array(predicts), np.array(features)


def train_fn()

def run_filter(model, train_files, test_files, filter_threshold=0.8):


def main():

    print("Load dataset")
    train_files = get_train_files("../UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/", 8, 2)
    test_train_files, test_files = get_test_files("../UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/", 8, 2)
    train_files += test_train_files

    print("Got %i train files, %i test files" % (len(train_files), len(test_files)))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean, std)

    train_dataset = VideoDataset(train_files + test_train_files,
                                 8,
                                 224,
                                 normalize_transform,
                                 'cpu',
                                 split='test',
                                 seed=2022)
    test_dataset = VideoDataset(train_files + test_train_files,
                                8,
                                224,
                                normalize_transform,
                                'cpu',
                                split='test', seed=2022)

    train_loader = DataLoader(train_dataset, batch_size=2,
                              shuffle=False,
                              num_workers=4)
    test_loader = DataLoader(train_dataset, batch_size=2,
                             shuffle=False,
                             num_workers=4)

    print([t['label'] for t in test_train_files[:20]])
    parser = utils.arg_parser()
    args = parser.parse_args(['--backbone_net', 'resnet', '-d', '50', '--temporal_module_name', 'TAM',
                              '--groups', '8'])
    args.num_classes = 2

    model, arch_name = build_model(args, test_mode=True)

    checkpoint = torch.load('./K400-TAM-ResNet-50-f32.pth', map_location='cpu')

    pretrained = checkpoint['state_dict']
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not (k == 'fc.weight' or k == 'fc.bias'):
            model_dict[k] = pretrained[k]


    model.load_state_dict(model_dict)
    model = model.to('cuda')

    run_filter(model, train_files, test_files, filter_threshold=0.8)
    train_output, train_fts = get_predictions(model, train_loader, device='cuda')
    print("Train snipsets: ", len(train_files + test_train_files))
    print(train_output.shape, train_fts.shape)

if __name__=="__main__":



# In[ ]:




