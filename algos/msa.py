import math
import numpy as np
from bench_utils import ObjectiveCounter, safe_clip

def levy_flight_step(D, beta=1.5):
    sigma_u = (
        math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(D) * sigma_u
    v = np.random.randn(D)
    v = np.where(np.abs(v) < 1e-12, 1e-12, v)
    step = u / (np.abs(v) ** (1 / beta))
    return np.clip(step, -10, 10)


def run_msa(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D = len(bounds)
    f = ObjectiveCounter(func, max_fes)
    X = np.array([[lb + np.random.rand()*(ub-lb) for (lb,ub) in bounds] for _ in range(pop_size)], dtype=float)
    fitness = np.array([f(x) for x in X], dtype=float)
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy(); best_val = fitness[best_idx]
    history = [best_val]

    while True:
        for i in range(pop_size):
            x_i = X[i].copy(); f_i = fitness[i]
            if np.random.rand() < 0.5:
                a, b = np.random.choice(pop_size, 2, replace=False)
                tau1 = levy_flight_step(D)
                tau2 = np.clip(np.random.randn(D), -5, 5)
                U = (np.random.rand(D) > 0.5).astype(float)
                x_new = x_i + (x_i - X[a]) * tau1 + np.abs(tau2) * U * (X[a] - X[b])
            else:
                mu = max(0.0, 1.0 - f.nfe / max_fes)
                d_s = -(x_i - best_sol)
                v_s = np.random.rand(D) * (1 - mu)
                x_new = x_i + v_s * d_s
            x_new = safe_clip(np.clip(x_new, -1e6, 1e6), bounds)
            try:
                f_new = f(x_new)
            except StopIteration:
                return best_sol, float(best_val), history
            if f_new < f_i:
                X[i] = x_new; fitness[i] = f_new
                if f_new < best_val:
                    best_val = f_new; best_sol = x_new.copy()
        history.append(float(best_val))