import cv2
import numpy as np
import albumentations as A
from albumentations import *


# define heavy augmentations
def get_training_augmentation_crop_image():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.Rotate(limit=3, p=0.5),
        A.OpticalDistortion(p=0.2),

        A.OneOf([
            CropNonEmptyMaskIfExists(256,320, p=0.3),
            A.RandomCrop(256, 320, p=0.7),
        ], p=1.0),
        A.PadIfNeeded(min_height=256, min_width=550, 
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, p=0.3),
        A.RandomCrop(256, 320, p=1.0)
    ]
    return A.Compose(train_transform)


def get_training_augmentation_whole_image():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.Rotate(limit=3, p=0.5),
        A.OpticalDistortion(p=0.2),

        A.PadIfNeeded(min_height=256, min_width=3000, 
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, p=0.3),
        A.RandomCrop(256, 1600, p=1.0)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
    ]
    return A.Compose(test_transform)


def get_preprocessing():
    _transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
    ]
    return A.Compose(_transform)
