from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from collections import namedtuple
from sklearn.metrics import make_scorer
from sklearn.metrics import auc
from scipy.stats import spearmanr
import numpy as np
import pandas as pd


"""
METRICAS

Attributes:
    Diccionario de métricas:

    METRICS = {
        "metric_key": (function, kwargs),
    }
"""


def spearman_rank_score(y_test, y_pred):
    coefs = []
    for yt, yp in zip(y_test, y_pred):
        coef, pval = spearmanr(yt, yp)
        coefs.append(coef)

    return np.mean(coefs)


def _accuracy_loss(y_score, y_pred):
    pred_scores = pd.DataFrame()
    for p in y_score[np.argsort(y_pred)]:
        pred_scores = pred_scores.append(p, ignore_index=True)

    pred_scores = pred_scores.T

    max_scores = pred_scores.max(axis=1)
    loss_scores = abs(pred_scores.sub(max_scores, axis=0)).cummin(axis=1)

    # nfs = loss_scores.shape[1]
    loss_auc = []
    for i in loss_scores.iterrows():
        loss_auc.append(auc(i[1].keys(), i[1].values))

    return np.mean(loss_auc)


def accuracy_loss(y_scores, y_pred):
    results = []
    for ys, yp in zip(y_scores, y_pred):
        results.append(_accuracy_loss(ys, yp))

    return np.mean(results)


def _mean_accuracy_loss(y_scores, y_pred, plot=False):
    max_score = y_scores.max()
    pred_scores = y_scores[np.argsort(y_pred)[::-1]]
    loss = max_score - np.maximum.accumulate(pred_scores)
    x = np.linspace(1, loss.shape[0], loss.shape[0])

    if plot:
        plt.plot(x, loss)

    return auc(x, loss)

def mean_accuracy_loss(y_score, y_pred):
    return np.mean([_mean_accuracy_loss(ys, yp)
                   for ys, yp in zip(y_score, y_pred)])




Metric = namedtuple("Metric", ["method", "kwargs", "name"])

METRICS = {
        'acc': Metric(accuracy_score, {},
                      "Precisión"),
        'gmean': Metric(geometric_mean_score, {"correction": 0.001},
                        "G-Mean"),
        'tpr': Metric(sensitivity_score, {'average': "micro"},
                      "Sensitividad"),
        'spc': Metric(specificity_score, {'average': "weighted"},
                      "Especificidad"),
        'spearman': Metric(spearman_rank_score, {},
                           "Spearman-Rank-Coefficient"),
        'acc_loss': Metric(accuracy_loss, {},
                           "Max Accuracy Loss"),
        'mean_acc_loss': Metric(mean_accuracy_loss, {},
                           "Mean Accuracy Loss"),
        }


def get_metric(metric_key):
    metric = METRICS[metric_key]
    return make_scorer(metric.method, **metric.kwargs)


def evaluate_metric(metric_key, y_true, y_pred):
    metric = METRICS[metric_key]
    return metric.method(y_true, y_pred, **metric.kwargs)
