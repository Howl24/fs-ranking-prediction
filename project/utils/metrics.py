from sklearn.metrics import accuracy_score
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from collections import namedtuple
from sklearn.metrics import make_scorer

"""
METRICAS

Attributes:
    Diccionario de métricas:

    METRICS = {
        "metric_key": (function, kwargs),
    }
"""

Metric = namedtuple("Metric", ["method", "kwargs", "name"])

METRICS = {
        'acc': Metric(accuracy_score, {}, "Precisión"),
        'gmean': Metric(geometric_mean_score, {"correction": 0.001}, "G-Mean"),
        'tpr': Metric(sensitivity_score, {'average': "micro"}, "Sensitividad"),
        'spc': Metric(specificity_score, {'average': "weighted"}, "Especificidad"),
        }

def get_metric(metric_key):
    metric = METRICS[metric_key]
    return make_scorer(metric.method, **metric.kwargs)

def evaluate_metric(metric_key, y_true, y_pred):
    metric = METRICS[metric_key]
    return metric.method(y_true, y_pred, **metric.kwargs)