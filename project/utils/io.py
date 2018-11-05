import pandas as pd
import os
import pickle
from collections import namedtuple
from project import CORPUS_PATH
import logging
import numpy as np

Dataset = namedtuple('Dataset', ['name', 'X', 'y'])


def get_corpus_path(corpus_key):
    return CORPUS_PATH[corpus_key]


def describe(dataset):
    info = {}
    info["dataset"] = dataset.name
    info["n_muestras"] = dataset.X.shape[0]
    info["n_atributos"] = dataset.X.shape[1]
    classes, counts = np.unique(dataset.y, return_counts=True)

    info["n_clases"] = len(classes)

    print("Dataset: \t\t", dataset.name)
    print("Nro de muestras: \t", dataset.X.shape[0])
    print("Nro de atributos: \t", dataset.X.shape[1])
    classes, counts = np.unique(dataset.y, return_counts=True)
    print("Nro de clases: \t\t", len(classes))
    print("Muestras por clase: ")
    for cls, count in zip(classes, counts):
        print("\t\t\t", cls, "->", count)

    return info


def read_dataset(filename, ext=".txt"):
    # Microarray file
    dataset = pd.read_csv(filename, sep=" ")
    dataset = dataset.transpose()
    X = dataset.drop(0, axis=1).values.astype("float")
    y = dataset[0].values
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]  # Get rid of extension
    return Dataset(name, X, y)


def read_datasets(corpus_key, ext=".txt"):
    path = get_corpus_path(corpus_key)

    filenames = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]

    datasets = []
    for filename in filenames:
        if filename.endswith(ext):
            datasets.append(read_dataset(filename))

    return datasets


def save_pickle(filename, data):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error("Archivo no encontrado.")
        return None


class ResultMixin(object):
    """Mixin generico de resultados.
    Atributos:
        - PATH: (string) Dirección en la que se guardará el resultado.
    """

    def save(self, filename=None):
        if filename:
            filename = self.PATH + filename
        else:
            filename = self.PATH + self.__str__()
        save_pickle(filename, self)

    def load(self, filename=None):
        if filename:
            filename = self.PATH + filename
        else:
            filename = self.PATH + self.__str__()
        obj = load_pickle(filename)
        return obj if obj else self
