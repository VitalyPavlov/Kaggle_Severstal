import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

class Metrics(Callback):
    
    def __init__(self, valid_dataloader):
        super().__init__()
        self.validation_data = valid_dataloader

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        
        pred, true = [], []
        
        for batch in range(batches):
            xVal, yVal = self.validation_data[batch]
            pred.extend(np.asarray(self.model.predict(xVal)).round())
            true.extend(yVal)

        pred = np.array(pred)
        true = np.array(true)

        _val_f1, _val_precision, _val_recall, _val_acc = [], [], [], []
        for i in range(4):
            _val_f1.append(round(f1_score(true[:,i], pred[:,i]), 3))
            _val_precision.append(round(precision_score(true[:,i], pred[:,i]), 3))
            _val_recall.append(round(recall_score(true[:,i], pred[:,i]), 3))
            _val_acc.append(round(accuracy_score(true[:,i], pred[:,i]), 3))

        val_f1 = round(np.mean(_val_f1), 3)

        logs['val_f1s'] = val_f1
        logs['val_f1'] = ' '.join(map(str, _val_f1))
        logs['val_presicion'] = ' '.join(map(str, _val_precision))
        logs['val_recall'] = ' '.join(map(str, _val_recall))
        logs['val_acc'] = ' '.join(map(str, _val_acc))

        print('f1_mean:', val_f1, 'f1:', _val_f1, 'presicion:', _val_precision, 'recall', _val_recall, 'acc', _val_acc)
        return
