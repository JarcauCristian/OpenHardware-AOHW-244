import logging
from time import time


def timeit(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        msg = f'Function {func.__name__!r} executed in {(t2-t1):.4f} s'
        print(msg)
        return result, round((t2-t1), 2)

    return wrap_func
