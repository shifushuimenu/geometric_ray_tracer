from time import time
from typing import Callable

__all__ = ["timer_func", "RayIntersectionNotFoundError"]

class RayIntersectionNotFoundError(AssertionError):
    pass

def timer_func(func : Callable) -> Callable:
    "Used as a decorator, this function shows the execution time of the function object passed."
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func