import numpy as np
import pandas as pd
import random
import os
import datetime

import time
import copy
import torch
from torchvision import models
import segmentation_models_pytorch as smp

from data_loader import *
from augmentation import *
from imblearn.over_sampling import SMOTE


BASE_DIR = '../../../input'
SEED = 2019
NUM_FOLD = 0
BATCH_SIZE = 16
LR = 4e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
REDUCE_LR_FACTOR = 0.25
REDUCE_LR_PATIENCE = 7
REDUCE_LR_MIN = 1e-6
PATH_WEIGTS = './weights_and_logs/model_fold_%d.h5'%NUM_FOLD
PATH_WEIGTS_PRETRAIN = './weights_and_logs/model_fold_%d.h5'%NUM_FOLD


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_metrics(true, pred):
    _pred = np.empty(shape=(4,256,1600))
    for i in range(5):
        _pred[:,:,i*320:i*320+320] = pred[i]
    
    dice_pos, dice_neg = [], []
    for i in range(_pred.shape[0]):
        p = _pred[i,:,:].reshape(-1,)
        t = true[i,:,:].reshape(-1,)

        if t.max() == 1:
            dice_pos.append((2 * (p * t).sum()) / (p.sum() + t.sum()))
            dice_neg.append(np.nan)
        else:
            dice_pos.append(np.nan)
            dice_neg.append(0 if p.max() == 1 else 1)

    return dice_pos, dice_neg


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, 
                scheduler, device, num_epochs=25, early_stop_patience=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = 0.0
    early_stoping = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0

        # Train part
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss_train = running_loss / dataset_sizes['train']
        
        # Valid part
        model.eval()
        optimizer.zero_grad()
        running_loss = 0.0
        
        dice_pos, dice_neg = [], []
        for inputs, labels in dataloaders['val']:
            inputs = inputs[0].to(device)
            labels = labels[0].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).long()
                
                _dice_pos, _dice_neg = get_metrics(labels.data.cpu().numpy(), 
                                                   preds.data.cpu().numpy())
                dice_pos.append(_dice_pos)
                dice_neg.append(_dice_neg)
            
        print('train loss: {:.4f}'.format(epoch_loss_train))
        
        metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
        print('val_dice:', np.round(metrics_val, 3), 
              'val_dice_pos:', np.round(np.nanmean(dice_pos, axis=0), 3), 
              'val_dice_neg:', np.round(np.nanmean(dice_neg, axis=0), 3))
        
        if metrics_val > best_metrics:
            print('*')
            best_metrics = metrics_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, PATH_WEIGTS)
            early_stoping = 0
        else:
            print()
            early_stoping += 1
        
        if early_stoping > early_stop_patience:
            break
            
        scheduler.step(metrics_val)
        print()
        
        with open('./weights_and_logs/logs.txt','a') as f:
            f.write('''train_loss: {:.4f} val_metrics: {:.4f}\n'''.format(epoch_loss_train, 
                                                                          metrics_val))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val metrics: {:4f}'.format(best_metrics))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    seed_everything(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(os.path.join('../../../input', 'train.csv'))
    train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

    folds_ids = pd.read_csv('train-5-folds.csv')
    train_files = folds_ids.loc[folds_ids.fold != NUM_FOLD, 'ImageId_ClassId'].values
    valid_files = folds_ids.loc[folds_ids.fold == NUM_FOLD, 'ImageId_ClassId'].values

    # Dataset for train images
    train_dataset = Dataset_train(
        ids=train_files, 
        df=train_df,
        augmentation=get_training_augmentation_crop_image(),
        preprocessing=get_preprocessing()
    )

    # Dataset for validation images
    valid_dataset = Dataset_valid(
        ids=valid_files, 
        df=train_df,
        preprocessing=get_preprocessing()
    )

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, shuffle=True, 
                                                        num_workers=0, batch_size=BATCH_SIZE),
                   'val': torch.utils.data.DataLoader(valid_dataset, shuffle=False, 
                                                      num_workers=0, batch_size=1)
                  }

    dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)*5}

    model = smp.Unet('efficientnet-b3', classes=4, activation=None, encoder_weights='imagenet')
    print(model)
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=LR)
    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', 
                                                                  factor=REDUCE_LR_FACTOR,
                                                                  patience=REDUCE_LR_PATIENCE,  
                                                                  min_lr=REDUCE_LR_MIN,
                                                                  verbose=True)
    
    #model.load_state_dict(torch.load(PATH_WEIGTS_PRETRAIN))
    model = train_model(model, criterion, optimizer_ft, dataloaders,
                        dataset_sizes, exp_lr_scheduler, device,
                        num_epochs=EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE)
    

if __name__=='__main__':
    main()
