import numpy as np
import pandas as pd
import random
import os
import datetime

import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.engine.training import Model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D, Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from rectified_adam import RectifiedAdam
from data_loader import *
from augmentation import *
from metrics import *
from rectified_adam import *
from imblearn.over_sampling import SMOTE


BASE_DIR = '../../../input'
SEED = 2019
NUM_FOLD = 0
BATCH_SIZE = 4
LR = 1e-5
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
    
    es_callback = EarlyStopping(monitor="val_f1s", patience=early_stop_patience, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_f1s', factor=reduce_lr_factor,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, 
                                  verbose=1, mode='max')
    
    csv_logger = CSVLogger('./weights_and_logs/log_fold_%d.csv'%NUM_FOLD, append=False, separator=';')

    mc_callback_best = ModelCheckpoint(weights_path, monitor='val_f1s', 
                                       verbose=0, save_best_only=True,
                                       save_weights_only=True, mode='max', period=1)
    
    callbacks = [metrics, es_callback, reduce_lr, csv_logger, mc_callback_best]
    return callbacks


def build_model(input_shape):
    input_tensor = Input(shape=input_shape)
    base = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    for l in base.layers:
        l.trainable = True
    
    conv = base.output

    x = GlobalAveragePooling2D(name='pool1')(conv)
    x = BatchNormalization()(x)
    x = Dense(512, name='fc1')(x)
    x = Activation('relu', name='relu1')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(4, name='fc3')(x)
    x = Activation('sigmoid', name='sigmoid')(x)
    model = Model(inputs=[base.input], outputs=[x])
    return model


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
        augmentation=get_training_augmentation_whole_image(),
        preprocessing=get_preprocessing()
    )

    # Dataset for validation images
    valid_dataset = Dataset_train(
        ids=valid_files, 
        df=train_df,
        preprocessing=get_preprocessing()
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    K.clear_session()
    model = build_model((256, 1600, 3))
    model.compile(optimizer=RectifiedAdam(lr=LR), loss='binary_crossentropy')
    #model.summary()
    
    metrics = Metrics(valid_dataset)
    checkpoint = get_callback(weights_path='./weights_and_logs/model_whole_image_fold_%d.h5'%NUM_FOLD, metrics=metrics)
    
    # train model
    model.load_weights('./weights_and_logs/model_fold_%d.h5'%NUM_FOLD)
    history = model.fit_generator(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=checkpoint,
    )
    
if __name__=='__main__':
    main()
