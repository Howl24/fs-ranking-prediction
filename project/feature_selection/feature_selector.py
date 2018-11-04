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
from sklearn.preprocessing import minmax_scale

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

    def _evaluate_cv(self, X, y, pipelines):
        """Evalúa los pipelines siguiendo el método de cv asignado
           X, y -> dataset
           results -> instancia de FSResults.
        """

        cv_results = {}
        for p in pipelines:
            cv_results[p.info()] = {}

        for fold_idx, (train, test) in enumerate(self.cv.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            if self.max_n_features is None:
                max_n_features_ = X_train.shape[1]
            else:
                max_n_features_ = self.max_n_features

            for pipeline in pipelines:
                pipeline_ = clone(pipeline)

                start_time = time.process_time()
                pipeline_.fit(X_train, y_train,
                              fs__n_features=max_n_features_)
                elapsed_time = time.process_time() - start_time

                ppl_results = []
                for n_features in range(1, max_n_features_ + 1):
                    pipeline_.fit(X_train, y_train,
                                  fs__n_features=n_features)


                    y_pred = pipeline_.predict(X_test)

                    # Instead of saving the score, we will save
                    # the predictions and probabilities
                    # so we can apply other metrics without 
                    # running again the experiments

                    if hasattr(pipeline_, "decision_function"):
                        y_prob = pipeline_.decision_function(X_test)
                    elif hasattr(pipeline, "predict_proba"):
                        y_prob = pipeline_.predict_proba()
                    else:
                        y_prob = y_pred

                    #score = evaluate_metric(self.metric, y_test, y_pred)
                    support_ = pipeline_.named_steps['fs'].support_
                    # Reduce feature array size
                    features = np.where(support_)[0].astype('int16')
                    ppl_results.append((features, y_pred, y_prob))

                cv_results[pipeline_.info()][fold_idx] = (elapsed_time,
                                                          ppl_results)

        return cv_results

    def evaluate_dataset(self, dataset, reset=False):
        X = dataset.X
        y = dataset.y

        ev_results = FSResults(dataset.name)

        if not reset:
            # Load previous dataset evaluation results
            ev_results = ev_results.load()

        new_pipelines = [p for p in self.pipelines
                         if p.info() not in ev_results.pipelines()]

        logger.info(" Evaluación de dataset: %s iniciada.", dataset.name)

        cv_results = self._evaluate_cv(X, y, new_pipelines)
        for pipeline_id in cv_results:
            ev_results.add_results(pipeline_id, cv_results[pipeline_id])

        ev_results.save()

        logger.info(" Evaluación de dataset: %s terminada.", dataset.name)
        return ev_results

    def run(self, corpus_id, reset=False, n_jobs=1):
        """Evalúa datasets a partir de un corpus id"""
        datasets = read_datasets(corpus_id)
        return self._run(corpus_id, datasets, reset, n_jobs)

    def _run(self, corpus_id, datasets, reset=False, n_jobs=1):
        """Evalúa un listado de datasets"""
        logger.info("Evaluando " + corpus_id)
        configurations = [(d, reset) for d in datasets]

        parallel(self.evaluate_dataset, configurations, n_jobs)

        # Load all results and return a Collection
        # results = FSRCollection(corpus_id)
        # for dataset in datasets:
        #     fsr = FSResults(dataset.name).load()
        #     results.add_result(fsr)

        # return results


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
            sorted_scores = sorted(self.scores.items(),
                                   key=lambda x: x[1])  # reverse=True
            ranking = {item: idx+1
                       for idx, (item, score) in enumerate(sorted_scores)}

        if rank_method == "max_float":
            fs_methods, scores = zip(*self.scores.items())
            rank_scores = minmax_scale(scores,
                                       feature_range=(1, len(fs_methods)))

            ranking = {item: score
                       for item, score in zip(fs_methods, rank_scores)}

        return ranking


class FSResults(ResultMixin):
    PATH = FEATURE_SELECTION_RESULT_PATH

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.results = {}  # results by pipeline

    def __str__(self):
        return "_".join([
            self.dataset_id,
        ])

    def pipelines(self):
        return self.results.keys()

    def add_results(self, pipeline_id, pipeline_results):
        self.results[pipeline_id] = pipeline_results

    def resultsToDataFrame(self, pipelines=None):
        data = []

        if not pipelines:
            pipelines = self.results.keys()

        for pipeline in pipelines:
            ppl_results = self.results[pipeline]
            for fold_idx in ppl_results:
                elapsed_time, result_list = ppl_results[fold_idx]
                for features, score in result_list:
                    data.append({
                        "fold": fold_idx,
                        "nf": len(features),
                        "score": score,
                        "pipeline": pipeline,
                        })
        return pd.DataFrame(data)

    def plot(self, **kwargs):

        if not kwargs:
            kwargs = {
                    "ci": [90],
                    }

        plt.figure(figsize=(20, 10))

        sns.tsplot(data=self.resultsToDataFrame(),
                   value="score",
                   condition="pipeline",
                   time='nf',
                   unit="fold",
                   **kwargs)  # ci=[90])
        plt.xlabel("Número de atributos")
        plt.ylabel('Score')

    def ranking(self, rank_method, fs_list=None):
        data = self.resultsToDataFrame(pipelines=fs_list)
        group = data.groupby(['pipeline', 'nf']).mean().T

        pipelines = group.columns.levels[0]
        self.rank = Ranking()
        for pipeline in pipelines:
            score = group[pipeline].loc['score']

            # Ranking based on max score in feature range
            max_score = score.max()
            # max_nf_score = score.idxmax()

            self.rank.add_score(pipeline, max_score)

        return self.rank.build_ranking(rank_method=rank_method)

    # ----------------------------------------------------------------------------

    def foldsToDataFrame(self):
        data = []
        for fold, (train, test) in enumerate(self.folds):
            for idx in train:
                data.append({
                    "fold": fold,
                    "type": "train",
                    "idx": idx,
                    })

            for idx in test:
                data.append({
                    "fold": fold,
                    "type": "test",
                    "idx": idx,
                    })

        df = pd.DataFrame(data).set_index(['fold', 'type'])
        return df

    def dataFrameToFolds(self, df):
        self.folds = []
        for fold in df.index.get_level_values('fold').unique():
            train_idx = df.loc[fold, 'train']['idx'].values
            test_idx = df.loc[fold, 'test']['idx'].values

            self.folds.append((train_idx, test_idx))
        return self.folds

    def toDataFrames(self):
        results_df = self.resultsToDataFrame()
        folds_df = self.foldsToDataFrame()

        return [('folds', folds_df),
                ('results', results_df),
                ]

    def copy(self, results=True):
        new_fsr = FSResults(self.dataset_id)
        for train, test in self.folds:
            new_fsr.add_fold(train, test)

        if results:
            for fold_idx, results in self.results.items():
                new_fsr.add_results(fold_idx, results)

        return new_fsr


class FSRCollection(ResultMixin):
    PATH = FEATURE_SELECTION_RESULT_PATH

    def __init__(self, evaluation_id):
        self.evaluation_id = evaluation_id
        self.results = {}

    def __str__(self):
        return "_".join([
            self.evaluation_id,
        ])

    def size(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results.values())

    def add_result(self, fs_result):
        self.results[fs_result.dataset_id] = fs_result

    def ranking(self, rank_method, return_scores=False,
                fs_list=None):

        df_rank = pd.DataFrame([{
                    "dataset": r.dataset_id,
                    **r.ranking(rank_method,
                                fs_list=fs_list),
                    } for d, r in self.results.items()])
        df_rank.set_index("dataset", inplace=True)

        df_scores = None
        if return_scores:
            df_scores = pd.DataFrame([{
                "dataset": r.dataset_id,
                **r.rank.scores,
                } for d, r in self.results.items()])
            df_scores.set_index("dataset", inplace=True)

        return df_rank, df_scores

    def copy(self, fsr_results=True):
        new_fsrc = FSRCollection(self.evaluation_id)

        for dataset_name, fs_results in self.results.items():
            new_fsr = fs_results.copy(results=fsr_results)
            new_fsrc.add_result(new_fsr)

        return new_fsrc

# -----------------------------------------------------------------------------
