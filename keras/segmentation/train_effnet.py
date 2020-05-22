import numpy as np
import pandas as pd
import random
import os

import tensorflow as tf
import segmentation_models as sm
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger

from data_loader import *
from augmentation import *
from metrics import *
from rectified_adam import *
from imblearn.over_sampling import SMOTE


BASE_DIR = '../../../input'
SEED = 2019
NUM_FOLD = 0
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 12
LR = 1e-3
EPOCHS = 100


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_callback(weights_path, metrics):
    early_stop_patience = 15
    reduce_lr_factor = 0.25
    reduce_lr_patience = 10
    reduce_lr_min = 1e-6
    
    es_callback = EarlyStopping(monitor="val_dice_coef", patience=early_stop_patience, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=reduce_lr_factor,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, 
                                  verbose=1, mode='max')
    
    csv_logger = CSVLogger('./weights_and_logs/log_fold_%d.csv'%NUM_FOLD, append=False, separator=';')

    mc_callback_best = ModelCheckpoint(weights_path, monitor='val_dice_coef', 
                                       verbose=0, save_best_only=True,
                                       save_weights_only=True, mode='max', period=1)
    
    callbacks = [metrics, es_callback, reduce_lr, csv_logger, mc_callback_best]
    return callbacks


def main():
    seed_everything(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    
    
    folds_ids = pd.read_csv('train-5-folds.csv')
    train_files = folds_ids.loc[folds_ids.fold != NUM_FOLD, 'ImageId_ClassId'].values
    valid_files = folds_ids.loc[folds_ids.fold == NUM_FOLD, 'ImageId_ClassId'].values

    # Dataset for train images
    train_dataset = Dataset_train(
        ids=train_files, 
        df=train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing()
    )

    # Dataset for validation images
    valid_dataset = Dataset_valid(
        ids=valid_files, 
        df=train_df,
        preprocessing=get_preprocessing()
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    K.clear_session()
    model = sm.Unet(BACKBONE, classes=4, activation='sigmoid', encoder_weights='imagenet')
    model.compile(optimizer=RectifiedAdam(lr=1e-3), loss=sm.losses.binary_focal_dice_loss)
    
    metrics = Metrics(valid_dataset)
    checkpoint = get_callback(weights_path='./weights_and_logs/model_fold_%d.h5'%NUM_FOLD, metrics=metrics)
    
    # train model
    #model.load_weights('model_ef_unet_radam_bce_augm.h5')
    history = model.fit_generator(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=checkpoint,
    )
    
    
if __name__=='__main__':
    main()
