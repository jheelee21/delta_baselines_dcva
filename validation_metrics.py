from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def validation(damage_label, cd_label):
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    acc_list = []

    for truth, pred in zip(damage_label, cd_label):
        cd_corrects = np.count_nonzero(truth == pred) / (256 * 256)
        acc_list.append(cd_corrects)
        
        ret = confusion_matrix(truth.flatten(), pred.flatten(), labels=[0, 1]).ravel()

        tn, fp, fn, tp = ret[0], ret[1], ret[2], ret[3]
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

    acc_mean = np.array(acc_list).mean()
    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    
    P = 0 if tp + fp == 0 else tp / (tp + fp)
    R = 0 if tp + fn == 0 else tp / (tp + fn)
    F1 = 0 if R + P == 0 else 2 * P * R / (R + P)

    return acc_mean, P, R, F1
            