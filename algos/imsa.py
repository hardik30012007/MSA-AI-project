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

def chaotic_initialization(bounds, pop_size, D):
    X = np.zeros((pop_size, D))
    z = np.random.rand()
    for i in range(pop_size):
        for d, (lb, ub) in enumerate(bounds):
            z = 4.0 * z * (1.0 - z)
            X[i, d] = lb + z * (ub - lb)
    return X

def run_imsa(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D = len(bounds)
    f = ObjectiveCounter(func, max_fes)
    X = chaotic_initialization(bounds, pop_size, D)
    fitness = np.array([f(x) for x in X], dtype=float)
    best_idx = np.argmin(fitness)
    best_sol = X[best_idx].copy(); best_val = fitness[best_idx]
    history = [best_val]
    t = 1
    max_iters = max(1, (max_fes - pop_size) // max(1, int(pop_size*1.15)))

    while True:
        p_t = 0.8 - 0.6 * min(1.0, t / max_iters)
        sorted_idx = np.argsort(fitness)
        elite_count = max(2, pop_size // 5)
        elite_indices = sorted_idx[:elite_count]

        for i in range(pop_size):
            x_i = X[i].copy(); f_i = fitness[i]
            if np.random.rand() < p_t:
                a, b = np.random.choice(pop_size, 2, replace=False)
                tau1 = levy_flight_step(D)
                tau2 = np.clip(np.random.randn(D), -5, 5)
                U = (np.random.rand(D) > 0.5).astype(float)
                x_new = x_i + (x_i - X[a]) * tau1 + np.abs(tau2) * U * (X[a] - X[b])
            else:
                elite_sol = X[np.random.choice(elite_indices)]
                d_best = best_sol - x_i
                d_elite = elite_sol - x_i
                w1 = 1.0 - min(1.0, t / max_iters)
                w2 = min(1.0, t / max_iters)
                rand_vec = np.random.rand(D)
                x_new = x_i + rand_vec * (w1 * d_elite + w2 * d_best)
            x_new = safe_clip(np.clip(x_new, -1e6, 1e6), bounds)
            try:
                f_new = f(x_new)
            except StopIteration:
                return best_sol, float(best_val), history
            if f_new < f_i:
                X[i] = x_new; fitness[i] = f_new
                if f_new < best_val:
                    best_val = f_new; best_sol = x_new.copy()

        # worst restart
        num_worst = max(1, pop_size // 10)
        worst_indices = np.argsort(fitness)[-num_worst:]
        for idx in worst_indices:
            if np.random.rand() < 0.3:
                new_x = np.array([lb + np.random.rand()*(ub-lb) for (lb, ub) in bounds], dtype=float)
                new_x = safe_clip(new_x, bounds)
                try:
                    new_f = f(new_x)
                except StopIteration:
                    return best_sol, float(best_val), history
                if new_f < fitness[idx] or np.random.rand() < 0.2:
                    X[idx] = new_x; fitness[idx] = new_f
                    if new_f < best_val:
                        best_val = new_f; best_sol = new_x.copy()
        history.append(float(best_val))
        t += 1