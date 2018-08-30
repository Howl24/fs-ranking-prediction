import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.base import SelectorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from project import FEATURE_SELECTION_RESULT_PATH
from project.utils import read_datasets
from project.utils.io import ResultMixin
from project.utils.metrics import evaluate_metric
from project.utils.utils import parallel
from project.feature_selection.methods import evaluate_feature_selection

import logging
logger = logging.getLogger("feature_selection")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('feature_selection.log')
ch = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(name)s " +
                              "- %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

if not len(logger.handlers):
    logger.addHandler(fh)
    logger.addHandler(ch)


class FSPipeline(Pipeline):
    """ Pipeline de selección de características.
    (Debe poseer un 'step' con id 'fs')
    """
    def info(self):
        data = []
        for step_id, step in self.steps:
            if step_id == "fs":
                data.append(step.fs_id)
            else:
                data.append(step_id)

        return "_".join(data)


class FSPipelineEvaluator():
    """ Evalúa un pipeline de selección de características.
    Atributos:
        pipelines: Lista de pipelines de selección de características.
        cv : Iterador de validación cruzada.
        metric: String que indica la métrica a utilizar.
        max_n_features: Integer indicando el máximo número
                        de características a seleccionar.
                        None -> (Todas)
    """

    def __init__(self, pipelines, cv, metric, max_n_features=None):
        self.pipelines = pipelines
        self.cv = cv
        self.metric = metric
        self.max_n_features = max_n_features

    def _evaluate_fold(self, X, y, train, test):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        if self.max_n_features is None:
            max_n_features_ = X_train.shape[1]
        else:
            max_n_features_ = self.max_n_features

        results = {}
        for pipeline in self.pipelines:
            # Fit the pipeline to get the selected features
            # from 1 to max_n_features
            pipeline_ = clone(pipeline)

            # Calculate feature selection time and set selected features
            start_time = time.process_time()
            pipeline_.fit(X_train, y_train,
                          fs__n_features=max_n_features_)
            elapsed_time = time.process_time() - start_time

            pipeline_results = []
            for n_features in range(1, max_n_features_ + 1):
                pipeline_.fit(X_train, y_train, fs__n_features=n_features)
                y_pred = pipeline_.predict(X_test)
                score = evaluate_metric(self.metric, y_test, y_pred)

                support_ = pipeline_.named_steps['fs'].support_
                # Reduce feature array size
                features = np.where(support_)[0].astype('int16')
                pipeline_results.append((features, score))

            results[pipeline_.info()] = (elapsed_time, pipeline_results)
        return results

    def evaluate_dataset(self, dataset):
        X = dataset.X
        y = dataset.y

        ev_results = FSResults(dataset.name)
        logger.info(" Evaluación de dataset: %s iniciada.", dataset.name)
        for train, test in self.cv.split(X, y):
            fold_results = self._evaluate_fold(X, y, train, test)
            ev_results.add_results(train, test, fold_results)
        logger.info(" Evaluación de dataset: %s terminada.", dataset.name)
        return ev_results

    def run(self, corpus_id, reset=False, n_jobs=1):
        # TODO
        # Add reset
        # (Load previous evaluations and get new ones)
        datasets = read_datasets(corpus_id)
        return self._run(corpus_id, datasets, reset, n_jobs)

    def _run(self, corpus_id, datasets, reset=False, n_jobs=1):
        self.results = FSRCollection(corpus_id)
        logger.info("Evaluando " + corpus_id)
        datasets_ = [(d, ) for d in datasets]
        all_results = parallel(self.evaluate_dataset, datasets_, n_jobs)
        for fs_result in all_results:
            self.results.add_result(fs_result)

        return self.results


class FeatureSelection(BaseEstimator, SelectorMixin):
    """Feature selector for rank based methods.

    Parameters:
        fs_id: A string indicating a feature selection method.
        max_n_features: An integer indicating the maximun
                        number of selected features.

    Attributes:
        selected_features_: dict
            Selected features from 1 to max_n_features
    """

    def __init__(self, fs_id, max_n_features):
        self.fs_id = fs_id
        self.max_n_features = max_n_features

    def fit(self, X, y=None, n_features=None):
        """Selects features using the indicated feature selection method.

        Obtains a dictionary of selected features from 1 to max_n_features.
        After that, sets support_ mask with the selected features
        related to n_features.

        Parameters:
            n_features: integer, optional
                Number of features to select (default: max_n_features)
        """

        if not n_features:
            n_features = self.max_n_features

        try:
            check_is_fitted(self, 'selected_features_')
        except NotFittedError:
            self.selected_features_ = evaluate_feature_selection(self.fs_id,
                                                                 X, y,
                                                                 n_features)

        self.support_ = np.zeros(X.shape[1], dtype=np.bool)
        self.support_[self.selected_features_[n_features]] = True

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')

        return self.support_


# -----------------------------------------------------------------------------
# Results


class Ranking(object):
    def __init__(self):
        self.scores = {}

    def add_score(self, item, score):
        self.scores[item] = score

    def build_ranking(self, rank_method="max"):
        ranking = {}

        if rank_method == "max":
            sorted_scores = sorted(self.scores.items(), key=lambda x: x[1],
                                   reverse=True)
            ranking = {item: idx
                       for idx, (item, score) in enumerate(sorted_scores)}

        return ranking


class FSResults(ResultMixin):
    PATH = FEATURE_SELECTION_RESULT_PATH

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.folds = []
        self.results = {}  # results by fold. Key == fold idx

    def add_results(self, train_idx, test_idx, fold_results):
        idx = len(self.folds)
        self.folds.append((train_idx, test_idx))
        self.results[idx] = fold_results

    def toDataFrame(self):
        data = []
        for idx, fold_results in self.results.items():
            for pipeline in fold_results:
                elapsed_time, result_list = fold_results[pipeline]
                for features, score in result_list:
                    data.append({
                        "fold": idx,
                        "nf": len(features),
                        "score": score,
                        "pipeline": pipeline,
                        })
        return pd.DataFrame(data)

    def ranking(self):
        data = self.toDataFrame()
        group = data.groupby(['pipeline', 'nf']).mean().T

        pipelines = group.columns.levels[0]
        ranking = Ranking()
        for pipeline in pipelines:
            score = group[pipeline].loc['score']

            # Ranking based on max score in feature range
            max_score = score.max()
            # max_nf_score = score.idxmax()

            ranking.add_score(pipeline, max_score)

        return ranking.build_ranking(rank_method="max")

    def plot(self):
        plt.figure(figsize=(20, 10))

        sns.tsplot(data=self.toDataFrame(),
                   value="score",
                   condition="pipeline",
                   time='nf',
                   unit="fold",
                   ci=[90])
        plt.xlabel("Número de atributos")
        plt.ylabel('Score')


class FSRCollection(ResultMixin):
    PATH = FEATURE_SELECTION_RESULT_PATH

    def __init__(self, evaluation_id):
        self.evaluation_id = evaluation_id
        self.results = []

    def __str__(self):
        return "_".join([
            self.evaluation_id,
        ])

    def size(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def add_result(self, fs_result):
        self.results.append(fs_result)

    def ranking(self):
        return pd.DataFrame([r.ranking() for r in self.results])


# -----------------------------------------------------------------------------