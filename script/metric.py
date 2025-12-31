import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix
)



def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average='macro', zero_division=0)
    sn = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    auprc = average_precision_score(y_true, y_prob, average='macro')

    # specificity (SP)
    cm = confusion_matrix(y_true, y_pred)
    tn = []
    fp = []
    for i in range(len(cm)):
        tni = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fpi = np.sum(np.delete(cm[:, i], i))
        tn.append(tni)
        fp.append(fpi)
    sp = np.mean(np.array(tn) / (np.array(tn) + np.array(fp) + 1e-12))
    return acc, sn, sp, pr, f1, mcc, auroc, auprc