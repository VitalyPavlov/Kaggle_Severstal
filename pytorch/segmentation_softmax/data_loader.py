import os
import cv2
import numpy as np
import pandas as pd
import torch


BASE_DIR = '../../../input/train'
N_CROP = 5
IN_DIM = (256, 1600)
OUT_DIM = (N_CROP, 256, 320)
IMG_CHANELS = 3
MASK_CHANELS = 4


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T


def build_masks(rles, input_shape):
    depth = len(rles)
    masks = np.zeros((*input_shape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, input_shape)
    
    return masks


# classes for data loading and preprocessing
class Dataset_train:    
    def __init__(
            self, 
            ids,
            df,
            images_dir=BASE_DIR, 
            dim=IN_DIM,
            augmentation=None, 
            preprocessing=None
    ):
        self.ids =ids
        self.df=df
        self.images_path = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.dim=dim
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_path[i], -1)
        
        #build mask
        mask_df = self.df[self.df['ImageId'] == self.ids[i]]
        rles = mask_df['EncodedPixels'].values
        mask = build_masks(rles, input_shape=self.dim)
        mask = np.asarray(mask, np.uint8)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        mask = np.append(mask, (1 - mask.max(axis=2)).reshape(256,320,1), axis=2)
        mask = np.argmax(mask, axis=2)

        return torch.Tensor(image).permute(2, 0, 1), torch.Tensor(mask).long()
        
    def __len__(self):
        return len(self.ids)
    

class Dataset_valid:    
    def __init__(
            self, 
            ids,
            df,
            images_dir=BASE_DIR, 
            dim=IN_DIM,
            augmentation=None, 
            preprocessing=None
    ):
        self.ids =ids
        self.df=df
        self.images_path = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.dim=dim
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        _image = cv2.imread(self.images_path[i], -1)

        #build mask
        mask_df = self.df[self.df['ImageId'] == self.ids[i]]
        rles = mask_df['EncodedPixels'].values
        mask = build_masks(rles, input_shape=self.dim)
        
        image = np.empty((*OUT_DIM, IMG_CHANELS))
        for i in range(N_CROP):
            image[i] = _image[:, i * OUT_DIM[2] : (i + 1) * OUT_DIM[2], :]
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return torch.Tensor(image).permute(0, 3, 1, 2), torch.Tensor(mask).permute(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)
