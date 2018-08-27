from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import fisher_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_SELECTION_METHODS = {
        'reliefF': reliefF.reliefF,
        'fisher': fisher_score.fisher_score,
        }


def evaluate_score(fs_key, X, y):
    fs_method = FEATURE_SELECTION_METHODS[fs_key]
    return fs_method(X, y)


class FeatureFilter(object):
    """ Filter features using different ranking methods"""

    def __init__(self, ranker_methods=None, score_threshold=0.8):
        if ranker_methods is None:
            ranker_methods = []

        self.ranker_methods = ranker_methods
        self.score_threshold = score_threshold

    def evaluate_rankers(self, dataset):
        scores = {fs: evaluate_score(fs, dataset.X, dataset.y)
                  for fs in self.ranker_methods}
        self.scores = pd.DataFrame(scores)

    def normalize(self):
        for fs in self.scores:
            self.scores[fs] = self.scores[fs] / np.sum(self.scores[fs])

    def filter_features(self):
        self.ffs = []
        for fs in self.scores:
            values = self.scores[fs].sort_values(ascending=False)
            cumsum = np.cumsum(values)
            ff = np.array(cumsum[cumsum < self.score_threshold].index)
            self.ffs.append(ff)

        return self.ffs

    def plot(self):
        for fs in self.scores:
            values = sorted(self.scores[fs], reverse=True)
            x = range(0, len(values))
            plt.plot(x, np.cumsum(values), label=fs)

        plt.legend()
        plt.show()
