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
        mf_data, fs_data = self._get_data()
        df_mf = mf_data.toDataFrame()
        df_rank, df_scores = fs_data.ranking("max", return_scores=True)

        return (df_mf.sort_index(),
                df_rank.sort_index(),
                df_scores.sort_index())

        # fold_data = []
        # for fsr in fs_data:
        #     group = fsr.resultsToDataFrame().groupby(['pipeline',
        #                                               'fold']).max().T
        #     pipelines = group.columns.levels[0]
        #     fold_scores = {p: group[p].loc['score'] for p in pipelines}

        #     fold_data.append({
        #         'dataset': fsr.dataset_id,
        #         **fold_scores,
        #         })

        # df_fold_scores = pd.DataFrame(fold_data).set_index('dataset')

        #return df_mf, df_rank, df_scores, df_fold_scores

    def _get_data(self):
        return MFRCollection(self.corpus_id).load(), FSRCollection(self.corpus_id).load()


    def fit(self, X, y):
        self._fit_y = y
        return super().fit(X)

    def predict(self, X):
        neighbors_idx = self.kneighbors(X, return_distance=False)
        neighbors_rank = self._fit_y[neighbors_idx]

        pred_rank = []
        for neighbors in neighbors_rank:
            rank_mean = np.mean(neighbors, axis=0)
            rank = np.argsort(np.argsort(rank_mean)) + 1

            pred_rank.append(rank)

        return np.array(pred_rank)


class RandomRankingPredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.rank_size = y.shape[1]
        return self

    def predict(self, X):
        n, _ = X.shape

        pred_rank = []
        for i in range(n):
            pred_rank.append(np.random.permutation(self.rank_size) + 1)

        return np.array(pred_rank)
