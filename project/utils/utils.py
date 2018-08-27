import time
from multiprocessing import Pool


def calculate_time_lapse(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    lapse = time.time() - start
    return lapse, result


def parallel(func, configurations, n_jobs=None):
    with Pool(n_jobs) as pool:
        return pool.starmap(func, configurations)
