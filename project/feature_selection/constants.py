from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

def rfe(X, y, estimator, **kwargs):
    MIN_N_FEATURES = 1
    STEP = 1
    selector = RFE(estimator, MIN_N_FEATURES, STEP, verbose=1)
    return selector.fit(X, y, **kwargs)

feature_selection_functions = {
        'reliefF': reliefF.reliefF,
        'fisher_score': fisher_score.fisher_score,
        'rfe': rfe,
        }
