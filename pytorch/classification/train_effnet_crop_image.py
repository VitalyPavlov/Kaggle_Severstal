import numpy as np
import pandas as pd
import random
import os
import datetime

import time
import copy
import torch
from torchvision import models

from data_loader import *
from augmentation import *
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


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


def smote_adataset(x_train, y_train):
    """ Oversampling """

    sm = SMOTE(random_state=SEED)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

    return x_train_res, y_train_res


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        self.effnet.fc = torch.nn.Sequential()
        self.bn0 = torch.nn.BatchNorm1d(num_features=1000)
        self.fc1 = torch.nn.Linear(1000, 512)
        self.act1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(num_features=512)
        self.fc2 = torch.nn.Linear(512, 4)
        self.act2 = torch.nn.Sigmoid()
        
    
    def forward(self, x):
        x = self.effnet(x)
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        
        return x
    
    
def get_metrics(true, pred):
    val_f1, val_precision, val_recall, val_acc = [], [], [], []
    for i in range(4):
        val_f1.append(round(f1_score(true[:,i], pred[:,i]), 3))
        val_precision.append(round(precision_score(true[:,i], pred[:,i]), 3))
        val_recall.append(round(recall_score(true[:,i], pred[:,i]), 3))
        val_acc.append(round(accuracy_score(true[:,i], pred[:,i]), 3))

    val_f1_mean = np.mean(val_f1)

    print('f1_mean: {:.3f}, f1: {}, presicion: {}, recall: {}, acc: {}'.format(
          val_f1_mean, val_f1, val_precision, val_recall, val_acc), end=' ')
    return val_f1_mean


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
                preds = (outputs > 0.5).long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss_train = running_loss / dataset_sizes['train']
        
        # Valid part
        model.eval()
        optimizer.zero_grad()
        running_loss = 0.0
        
        pred, true = [], []
        for inputs, labels in dataloaders['val']:
            inputs = inputs[0].to(device)
            labels = labels[0].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                preds = (outputs > 0.5).long()
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            true.extend(labels.data.cpu().numpy())
            pred.extend(preds.data.cpu().numpy())
            
        epoch_loss_val = running_loss / dataset_sizes['val']
        
        print('train loss: {:.4f} val loss: {:.4f}'.format(epoch_loss_train, epoch_loss_val))
        
        pred = np.array(pred)
        true = np.array(true)
        
        metrics_val = get_metrics(true, pred)
        
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
        
        with open('./weights_and_logs/logs_4.txt','a') as f:
            f.write('''train_loss: {:.4f} val_loss: {:.4f} val_metrics: {:.4f}\n'''.format(epoch_loss_train, 
                                                                                           epoch_loss_val, 
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
    # train_files = folds_ids.loc[folds_ids.fold != NUM_FOLD, 'ImageId_ClassId'].values
    # valid_files = folds_ids.loc[folds_ids.fold == NUM_FOLD, 'ImageId_ClassId'].values

    # in case oversampling
    train_index = np.array(folds_ids.loc[folds_ids.fold != NUM_FOLD].index).reshape(-1, 1)
    train_group = folds_ids.loc[folds_ids.fold != NUM_FOLD, 'class'].values

    train_index, train_group = smote_adataset(train_index, train_group)
    train_index = train_index.ravel()
    train_files = folds_ids.loc[train_index, 'ImageId_ClassId'].values
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

    model = Net()
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=LR)
    onplateau_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', 
                                                                      factor=REDUCE_LR_FACTOR,
                                                                      patience=REDUCE_LR_PATIENCE,  
                                                                      min_lr=REDUCE_LR_MIN,
                                                                      verbose=True)
    #exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    
    lr_scheduler = onplateau_lr_scheduler
    
    
    #model.load_state_dict(torch.load(PATH_WEIGTS_PRETRAIN))
    model = train_model(model, criterion, optimizer_ft, dataloaders,
                        dataset_sizes, lr_scheduler, device,
                        num_epochs=EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE)
    

if __name__=='__main__':
    main()
