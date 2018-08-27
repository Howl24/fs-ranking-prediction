from . import MetaFeature as MF
from project import META_FEATURE_RESULT_PATH
import pandas as pd
from project.utils import read_datasets
from project.utils.preprocessing import evaluate_preprocessing
from project.utils.io import ResultMixin


class MFResult(ResultMixin):
    """Resultados de la extracción de meta-atributos
       sobre un conjunto de datos."""

    PATH = META_FEATURE_RESULT_PATH

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.metafeatures = {}

    def __str__(self):
        return "_".join([self.dataset_name])

    def add_metafeature(self, metafeature, value):
        self.metafeatures[str(metafeature)] = value

    def contains(self, metafeature):
        return str(metafeature) in self.metafeatures


class MFRCollection(ResultMixin):

    PATH = META_FEATURE_RESULT_PATH

    def __init__(self, corpus_key):
        self.corpus_key = corpus_key
        self.results = {}

    def __str__(self):
        return "_".join([self.corpus_key])

    def add_result(self, mf_result):
        self.results[mf_result.dataset_name] = mf_result

    def contains(self, dataset_name):
        return dataset_name in self.results

    def get(self, dataset_name):
        return self.results[dataset_name]

    def normalize(self, norm_key):
        df = self.toDataFrame()
        for column in df.columns:
            df[column] = evaluate_preprocessing(norm_key, df[column])
        return df

    def dataset_names(self):
        return [mf_result.dataset_name for mf_result in self.results]

    def toDataFrame(self, names=True):
        data = []
        for dataset, mf_result in self.results.items():
            data.append({"dataset": dataset,
                         **mf_result.metafeatures})

        df = pd.DataFrame(data)
        if names:
            df = df.set_index("dataset")
        else:
            df = df.drop(["dataset"], axis=1)

        return df


class MetaFeatureExtractor():

    def __init__(self, metafeatures):
        self.metafeatures = metafeatures
        self.process_dict = self.generate_process_dict(self.metafeatures)
        self.results = []

    def generate_process_dict(self, metafeatures):
        """Retorna un diccionario con la siguiente estructura:
            key   -> tipo de objeto a utilizar
            value -> procesamiento a realizar al objeto (diccionario):

                key     -> tipo de funcion a utilizar
                value   -> post procesamiento a realizar al resultado
                           de aplicar la funcion al objeto (diccionario):

                    key -> tipo de post procesamiento a utilizar
                    value -> meta feature asignado al proceso
        """

        proc_dict = {}
        for mf in metafeatures:
            if mf.obj_key not in proc_dict:
                proc_dict[mf.obj_key] = {}

            obj_dict = proc_dict[mf.obj_key]
            if mf.func_key not in obj_dict:
                obj_dict[mf.func_key] = {}

            func_dict = obj_dict[mf.func_key]
            if mf.post_key not in func_dict:
                func_dict[mf.post_key] = mf

        return proc_dict

    def _run(self, corpus_key, datasets, reset=False):
        processing_dict = self.generate_process_dict(self.metafeatures)

        self.results = MFRCollection(corpus_key)

        if not reset:
            self.results = self.results.load()

        for dataset in datasets:
            if self.results.contains(dataset.name):
                prev = self.results.get(dataset.name)
                new_metafeatures = [mf for mf in self.metafeatures
                                    if str(mf) not in prev.metafeatures]
                proc_dict = self.generate_process_dict(new_metafeatures)

                mf_result = self.process_dataset(dataset, proc_dict, prev)
            else:
                mf_result = self.process_dataset(dataset, processing_dict)

            self.results.add_result(mf_result)

        return self.results

    def run(self, corpus_key, reset=False):
        datasets = read_datasets(corpus_key)
        return self._run(corpus_key, datasets, reset)

    def process_dataset(self, dataset, processing_dict, prev_result=None):
        if prev_result:
            mf_result = prev_result
        else:
            mf_result = MFResult(dataset.name)

        for obj_key in processing_dict:
            get_object = MF.get_object_method(obj_key)
            obj = get_object(dataset)
            meta_funct_processing_dict = processing_dict[obj_key]

            for meta_funct_key in meta_funct_processing_dict:
                meta_funct = MF.get_meta_function_method(meta_funct_key)
                processed_obj = meta_funct(obj)
                post_proc_dict = meta_funct_processing_dict[meta_funct_key]

                for post_proc_key in post_proc_dict:
                    post_process = MF.get_post_processing_method(post_proc_key)
                    post_processed_obj = post_process(processed_obj)
                    metafeature = post_proc_dict[post_proc_key]
                    mf_result.add_metafeature(metafeature, post_processed_obj)

        return mf_result
