import numpy as np
from keras.callbacks import Callback


def dice_coef(y_true, y_pred):
    dice_pos, dice_neg = [], []
    for i in range(4):
        p = y_pred[:,:,i].reshape(-1,)
        t = y_true[:,:,i].reshape(-1,)

        if t.max() == 1:
            dice_pos.append((2 * (p * t).sum()) / (p.sum() + t.sum()))
            dice_neg.append(np.nan)
        else:
            dice_pos.append(np.nan)
            dice_neg.append(0 if p.max() == 1 else 1)

    return dice_pos, dice_neg


class Metrics(Callback):
    
    def __init__(self, valid_dataloader):
        super().__init__()
        self.validation_data = valid_dataloader

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)

        dice_pos, dice_neg = [], []
        for batch in range(batches):
            x, y = self.validation_data[batch]
            _pred = np.asarray(self.model.predict(x)).round()

            pred = np.empty(shape=(256,1600,4))
            for i in range(5):
                pred[:,i*320:i*320+320,:] = _pred[i]

            #y = np.expand_dims(y, axis=0)
            #pred = np.expand_dims(pred, axis=0)

            _dice_pos, _dice_neg = dice_coef(y, pred)
            dice_pos.append(_dice_pos)
            dice_neg.append(_dice_neg)

        dice_pos = np.around(np.nanmean(dice_pos, axis=0), 3)
        dice_neg = np.around(np.nanmean(dice_neg, axis=0), 3)
        dice = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
        dice = round(dice, 3)

        logs['val_dice_coef'] = dice
        logs['val_dice_pos'] = ' '.join(map(str, dice_pos)) 
        logs['val_dice_neg'] = ' '.join(map(str, dice_neg)) 

        print('val_dice:', dice, 'val_dice_pos:', dice_pos, 'val_dice_neg:', dice_neg)
        return
