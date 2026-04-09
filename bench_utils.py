import math
import numpy as np
from dataclasses import dataclass

@dataclass
class ObjectiveCounter:
    func: callable
    max_fes: int
    nfe: int = 0

    def __call__(self, x):
        if self.nfe >= self.max_fes:
            raise StopIteration
        self.nfe += 1
        val = self.func(np.asarray(x, dtype=float))
        if np.isnan(val) or np.isinf(val):
            return 1e308
        return float(val)


def safe_clip(x, bounds):
    x = np.asarray(x, dtype=float).copy()
    for i, (lb, ub) in enumerate(bounds):
        if np.isnan(x[i]) or np.isinf(x[i]):
            x[i] = lb + np.random.rand() * (ub - lb)
        x[i] = np.clip(x[i], lb, ub)
    return x


def eval_budget_to_iters(max_fes, pop_size, init_extra=0):
    usable = max(1, max_fes - init_extra)
    return max(1, usable // pop_size)


def summarize_runs(values):
    arr = np.asarray(values, dtype=float)
    return float(np.min(arr)), float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr)>1 else 0.0), float(np.max(arr))
