from sklearn.neighbors import NearestNeighbors
from project.meta_feature.extractor import MFRCollection
from project.feature_selection.feature_selector import FSRCollection
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


class RankingPredictor(NearestNeighbors):
    def __init__(self, corpus_id, n_neighbors):
        super().__init__(n_neighbors)
        self.corpus_id = corpus_id

    def get_data(self):
        df_mf = MFRCollection(self.corpus_id).load().toDataFrame()

        fs_data = FSRCollection(self.corpus_id).load()
        df_rank, df_scores = fs_data.ranking("max", return_scores=True)


        fold_data =  []
        for fsr in fs_data:
            group = fsr.resultsToDataFrame().groupby(['pipeline',
                                                      'fold']).max().T
            pipelines = group.columns.levels[0]
            fold_scores = {p: group[p].loc['score'] for p in pipelines}

            fold_data.append({
                'dataset': fsr.dataset_id,
                **fold_scores,
                })
            
        df_fold_scores = pd.DataFrame(fold_data).set_index('dataset')

        return df_mf, df_rank, df_scores, df_fold_scores

    def fit(self, X, y):
        self._fit_y = y
        return super().fit(X)

    def predict(self, X):
        neighbors_idx = self.kneighbors(X, return_distance=False)
        neighbors_rank = self._fit_y[neighbors_idx]

        pred_rank = []
        for neighbors in neighbors_rank:
            rank_mean = np.mean(neighbors,axis=0)
            rank = np.argsort(np.argsort(rank_mean)) + 1

            pred_rank.append(rank)

        return np.array(pred_rank)


class RandomRankingPredictor(BaseEstimator):
    def __init__(self, rank_size):
        self.rank_size = rank_size

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        pred_rank = []
        for i in range(n_samples):
            pred_rank.append(np.random.permutation(self.rank_size) + 1)

        return np.array(pred_rank)
