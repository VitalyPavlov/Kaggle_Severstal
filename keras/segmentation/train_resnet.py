import numpy as np
import pandas as pd
import random
import os

import tensorflow as tf
import segmentation_models as sm
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from keras.applications.resnet50 import ResNet50
from keras.engine.training import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization

from data_loader import *
from augmentation import *
from metrics import *
from rectified_adam import *
from imblearn.over_sampling import SMOTE


BASE_DIR = '../../../input'
SEED = 2019
NUM_FOLD = 0
BATCH_SIZE = 12
LR = 1e-3
EPOCHS = 100


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def build_model(input_shape):
    input_tensor = Input(shape=input_shape) 
    resnet_base = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    #resnet_base.summary()
    
    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = UpSampling2D()(conv9)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(4, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(inputs=[resnet_base.input], outputs=[x])
    return model


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
    model = build_model((256, 320, 3))
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
