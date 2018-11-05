from . import constants
import logging
import numpy as np
import scipy.stats
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


class MetaFeature():
    """Esta clase almacena las claves de los tres componentes de un meta-atributo
    (objeto, función y post-procesamiento)."""

    object_methods = {}
    meta_function_methods = {}
    post_processing_methods = {}

    def __init__(self, obj_key, func_key, post_key):
        self.obj_key = obj_key
        self.func_key = func_key
        self.post_key = post_key

    def __str__(self):
        return "_".join([self.obj_key,
                         self.func_key,
                         self.post_key])

    @classmethod
    def get_object_method(cls, key):
        try:
            return cls.object_methods[key]
        except KeyError:
            return None

    @classmethod
    def get_meta_function_method(cls, key):
        try:
            return cls.meta_function_methods[key]
        except KeyError:
            return None

    @classmethod
    def get_post_processing_method(cls, key):
        try:
            return cls.post_processing_methods[key]
        except KeyError:
            return None

    def extract(self, dataset):
        obj = self.object_methods[self.obj_key](dataset.X, dataset.y)
        proc_obj = self.meta_function_methods[self.func_key](obj)
        post_proc_obj = self.post_processing_methods[self.post_key](proc_obj)
        return post_proc_obj

    @classmethod
    def MapMethods(cls):
        cls.object_methods = {
            constants.OBJECT_X: cls.get_x,
            constants.OBJECT_Y: cls.get_y,
            constants.OBJECT_X_STD: cls.get_x_std,
            constants.OBJECT_X_MINMAX: cls.get_x_minmax,
            }

        cls.meta_function_methods = {
            constants.META_FUNCTION_NUM_OBS: cls.num_obs,
            constants.META_FUNCTION_NUM_ATTRS: cls.num_attrs,
            constants.META_FUNCTION_NUM_CLASSES: cls.num_classes,
            constants.META_FUNCTION_RATIO_OBS_ATTRS: cls.ratio_obs_attrs,
            constants.META_FUNCTION_STAND_DEV: cls.stand_dev,
            constants.META_FUNCTION_VAR_COEF: cls.var_coef,
            constants.META_FUNCTION_COVARIANCE: cls.covariance,
            constants.META_FUNCTION_CORRELATION: cls.correlation,
            constants.META_FUNCTION_SKEWNESS: cls.skewness,
            constants.META_FUNCTION_KURTOSIS: cls.kurtosis,
            constants.META_FUNCTION_EXP_VAR: cls.exp_var,
            constants.META_FUNCTION_NORM_CLASS_ENTROPY:
                cls.normalized_class_entropy,
            }

        cls.post_processing_methods = {
            constants.POST_PROC_NONE: cls.no_post,
            constants.POST_PROC_MEAN: cls.mean,
            constants.POST_PROC_MIN: cls.min,
            constants.POST_PROC_MAX: cls.max,
            constants.POST_PROC_N_T80_CUMSUM: cls.n_elem_t80_cumsum,
            constants.POST_PROC_N_T90_CUMSUM: cls.n_elem_t90_cumsum,
            }

    # ---------------------------------------------------------------
    # Métodos para la obtención de objetos

    @staticmethod
    def get_x(dataset):
        return dataset.X

    @staticmethod
    def get_y(dataset):
        return dataset.y

    @staticmethod
    def get_x_std(dataset):
        return scale(dataset.X)

    @staticmethod
    def get_x_minmax(dataset):
        return minmax_scale(dataset.X)

    # ---------------------------------------------------------------
    # Métodos para el procesamiento de objetos

    # Simples

    @staticmethod
    def num_obs(X):
        try:
            return X.shape[0]
        except (AttributeError, IndexError):
            logging.exception("No se pudo obtener el número de observaciones" +
                              "Formato de objeto incorrecto")

    @staticmethod
    def num_attrs(X):
        try:
            return X.shape[1]
        except (AttributeError, IndexError):
            logging.exception("No se pudo obtener el número de atributos." +
                              "Formato de objeto incorrecto")

    @staticmethod
    def num_classes(y):
        return len(np.unique(y))

    @staticmethod
    def ratio_obs_attrs(X):
        try:
            return X.shape[1] / X.shape[0]
        except (AttributeError, IndexError):
            logging.exception(
                    "No se pudo obtener el ratio entre observaciones" +
                    "y atributos. Formato de objeto incorrecto.")

    # Estadístiscos
    @staticmethod
    def stand_dev(X):
        """Calcula la desviación estándar a partir de una matrix numpy"""
        return np.std(X, ddof=0, axis=0)

    @staticmethod
    def var_coef(X):
        return MetaFeature.stand_dev(X)/MetaFeature.mean(X)

    @staticmethod
    def covariance(X_std):
        """Calcula la covarianza entre todos los pares de atributos."""
        cov_mat = np.cov(X_std, ddof=0, rowvar=False)
        keep = np.invert(np.tril(np.ones(cov_mat.shape)).astype('bool'))
        return cov_mat[keep]

    @staticmethod
    def correlation(X_std):
        """Calcula la correlación entre todos los pares de atributos."""
        corr_mat = np.corrcoef(X_std.T)
        keep = np.invert(np.tril(np.ones(corr_mat.shape)).astype('bool'))
        return corr_mat[keep]

    @staticmethod
    def skewness(X):
        return scipy.stats.skew(X, axis=0, bias=True)

    @staticmethod
    def kurtosis(X):
        return scipy.stats.kurtosis(X, axis=0, bias=True)

    @staticmethod
    def frac1d_coef(X_std):
        cov_mat = np.cov(X_std.T, ddof=0)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        return np.absolute(eig_vals.max()) / np.trace(cov_mat)

    @staticmethod
    def exp_var(X):
        # cov_mat = np.cov(X_std, rowvar=False, ddof=0)
        # eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])
        #               for i in range(len(eig_vals))]
        # eig_pairs.sort()
        # eig_pairs.reverse()
        # tot = sum(eig_vals)
        # var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

        pca = PCA(svd_solver="full")
        pca.fit(X)
        ev = pca.explained_variance_ratio_
        return ev

    # Basados en teoría de la información
    @staticmethod
    def normalized_class_entropy(y):
        values, counts = np.unique(y, return_counts=True)
        return scipy.stats.entropy(counts, base=len(values))

    # ---------------------------------------------------------------
    # Métodos de post procesamiento
    @staticmethod
    def no_post(proc):
        return proc

    @staticmethod
    def mean(proc):
        return np.mean(proc)

    @staticmethod
    def min(proc):
        return np.min(proc)

    @staticmethod
    def max(proc):
        return np.max(proc)

    @staticmethod
    def n_elem_threshold_cumsum(proc, t):
        cs = np.cumsum(sorted(proc, reverse=True))
        return len(cs[cs <= t])

    @staticmethod
    def n_elem_t80_cumsum(proc):
        return MetaFeature.n_elem_threshold_cumsum(proc, 0.8)

    @staticmethod
    def n_elem_t90_cumsum(proc):
        return MetaFeature.n_elem_threshold_cumsum(proc, 0.9)
