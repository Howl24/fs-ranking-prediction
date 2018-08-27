from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import CFS
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from project.feature_selection.genetic_algorithm import Individual
from project.feature_selection.genetic_algorithm import GeneticAlgorithm
from project.feature_selection.genetic_algorithm import tan_fitness_function
import numpy as np
from project.utils import calculate_time_lapse


def reliefF_evaluation(X, y, n_features, **kwargs):
    time_lapse, score = calculate_time_lapse(reliefF.reliefF,
                                             X, y)
    idx = np.argsort(score, 0)[::-1]

    selected_features = {}
    for nf in range(1, n_features+1):
        selected_features[nf] = idx[0:nf]

    return selected_features


def fisher_evaluation(X, y, n_features, **kwargs):
    time_lapse, score = calculate_time_lapse(fisher_score.fisher_score,
                                             X, y)
    idx = np.argsort(score, 0)[::-1]
    selected_features = {}

    for nf in range(1, n_features+1):
        selected_features[nf] = idx[0:nf]

    return selected_features


def chi_square_evaluation(X, y, n_features):
    time_lapse, score = calculate_time_lapse(chi_square.chi_square,
                                             X, y)
    idx = np.argsort(score, 0)[::-1]

    selected_features = {}
    for nf in range(1, n_features+1):
        selected_features[nf] = idx[0:nf]

    return selected_features


def cfs_evaluation(X, y, n_features):
    idx = CFS.cfs(X, y)

    selected_features = {}
    for nf in range(1, n_features + 1):
        selected_features[nf] = idx[0:nf]

    return selected_features


def _rfe_evaluation(X, y, n_features, estimator):
    MIN_N_FEATURES = 1
    STEP = 1
    selector = RFE(estimator, MIN_N_FEATURES, STEP)
    selector = selector.fit(X, y)
    ranking = selector.ranking_

    selected_features = {}
    for nf in range(1, n_features+1):
        selected_features[nf] = np.where(ranking <= nf)[0]

    return selected_features


def rfe_svc_evaluation(X, y, n_features):
    estimator = LinearSVC()
    return _rfe_evaluation(X, y, n_features, estimator)


def reliefF_ga_evaluation(X, y, n_features, **kwargs):
    return hybrid_ga_evaluation(X, y, n_features, ranker_key="reliefF",
                                **kwargs)


def fisher_ga_evaluation(X, y, n_features, **kwargs):
    return hybrid_ga_evaluation(X, y, n_features, ranker_key="fisher",
                                **kwargs)


FEATURE_SELECTION_METHODS = {
        'reliefF': reliefF_evaluation,
        'fisher': fisher_evaluation,
        'chi_square': chi_square_evaluation,
        'cfs': cfs_evaluation,
        'rfe_svc': rfe_svc_evaluation,
        'reliefF_ga': reliefF_ga_evaluation,
        'fisher_ga': fisher_ga_evaluation,
        }


def evaluate_feature_selection(fs_key, X, y, n_features, **kwargs):
    fs_method = FEATURE_SELECTION_METHODS[fs_key]
    return fs_method(X, y, n_features, **kwargs)


def hybrid_ga_evaluation(X, y, n_features, ranker_key, **kwargs):
    selected_features = evaluate_feature_selection(ranker_key, X, y,
                                                   n_features, **kwargs)
    features = selected_features[n_features]

    return ga_evaluation(X, y, features, **kwargs)


def ga_evaluation(X, y, features, ff_key, model, metric, cv,
                  population_size=40, generations=30,
                  acc_weight=0.9, plot=False):

    if ff_key == "tan":
        ff = lambda dna: tan_fitness_function(dna, X, y, features, model,
                                              metric, cv,
                                              acc_weight=acc_weight)

    Individual.Configure(dna_size=len(features), fitness_function=ff)
    ga = GeneticAlgorithm(population_size=population_size,
                          generations=generations)
    ga.run(plot=plot)

    # Fittest attribute saves fittest individual by nf
    return {nf: features[ind.dna] for nf, ind in ga.ranking.items()}
