from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
import logging

"""
PRE-PROCESAMIENTOS

Attributes:
    Diccionario de pre-procesamientos:

    PREPROCESSING = {
        "proc_key": (function, kwargs),
    }
"""


PREPROCESSING = {
    'minmax': (minmax_scale, {}),
    'std': (scale, {}),
    'none': (lambda x: x, {}),
}

def evaluate_preprocessing(proc_key, values):
    try:
        preprocessing, kwargs = PREPROCESSING[proc_key]
    except KeyError:
        msg = 'Clave de preprocesamiento "{0}" no encontrada'.format(proc_key)
        logging.exception(msg)
        return None

    return preprocessing(values, **kwargs)
