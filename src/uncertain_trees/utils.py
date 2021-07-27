import functools
import time
import io
import sys

import numpy as np


def nostdout(func):
    """Excludes the stdout of a function"""

    @functools.wraps(func)
    def wrapper_nostdout(*args, **kwargs):
        verbose = args[0].verbose
        if not verbose:
            save_stdout = sys.stdout
            sys.stdout = io.StringIO()
        value = func(*args, **kwargs)
        if not verbose:
            sys.stdout = save_stdout
        return value

    return wrapper_nostdout


def timeit(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        if args[0].verbose:
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def print_input_size(func):
    """Prints the size of the input array of a function"""

    @functools.wraps(func)
    def wrapper_print_input_size(*args, **kwargs):
        if args[0].verbose:
            print(f"Running {func.__name__!r} with {args[1].shape[0]} data points")
        value = func(*args, **kwargs)
        return value

    return wrapper_print_input_size


def binary_entropy(p: np.ndarray) -> np.ndarray:
    entropy_idx = (p[:, 1] != 1) & (p[:, 1] != 0)
    entropy = np.ones_like(p)
    pp = p[entropy_idx, :]
    entropy[entropy_idx, :] = -pp * np.log2(pp) - (1 - pp) * np.log2(1 - pp)
    return entropy


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))
