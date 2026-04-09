import math
import numpy as np
from bench_utils import ObjectiveCounter, safe_clip

# ---------- Shared ----------
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

# ---------- Base MSA standardized ----------
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

# ---------- Improved MSA standardized ----------
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

# ---------- Baselines ----------
def _init_pop(bounds, pop_size):
    return np.array([[lb + np.random.rand()*(ub-lb) for (lb,ub) in bounds] for _ in range(pop_size)], dtype=float)


def run_pso(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D = len(bounds); f = ObjectiveCounter(func, max_fes)
    X = _init_pop(bounds, pop_size); V = np.zeros((pop_size, D))
    fit = np.array([f(x) for x in X])
    pbest = X.copy(); pfit = fit.copy()
    gidx = np.argmin(fit); gbest = X[gidx].copy(); gfit = fit[gidx]
    hist=[gfit]
    while True:
        w=0.7; c1=1.5; c2=1.5
        for i in range(pop_size):
            r1=np.random.rand(D); r2=np.random.rand(D)
            V[i]=w*V[i]+c1*r1*(pbest[i]-X[i])+c2*r2*(gbest-X[i])
            X[i]=safe_clip(X[i]+V[i], bounds)
            try: val=f(X[i])
            except StopIteration: return gbest, float(gfit), hist
            fit[i]=val
            if val<pfit[i]: pfit[i]=val; pbest[i]=X[i].copy()
            if val<gfit: gfit=val; gbest=X[i].copy()
        hist.append(float(gfit))


def run_gwo(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X])
    hist=[]; t=0; max_iters=max(1,(max_fes-pop_size)//pop_size)
    while True:
        idx=np.argsort(fit)
        alpha,beta,delta=X[idx[0]],X[idx[1]],X[idx[2]]
        afit=fit[idx[0]]; hist.append(float(afit))
        a=2-2*(t/max_iters)
        for i in range(pop_size):
            A1=2*a*np.random.rand(D)-a; C1=2*np.random.rand(D)
            A2=2*a*np.random.rand(D)-a; C2=2*np.random.rand(D)
            A3=2*a*np.random.rand(D)-a; C3=2*np.random.rand(D)
            X1=alpha-A1*np.abs(C1*alpha-X[i])
            X2=beta-A2*np.abs(C2*beta-X[i])
            X3=delta-A3*np.abs(C3*delta-X[i])
            Xn=safe_clip((X1+X2+X3)/3.0,bounds)
            try: fit[i]=f(Xn)
            except StopIteration: return alpha.copy(), float(afit), hist
            X[i]=Xn
        t+=1


def run_de(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X]); hist=[]
    F=0.5; CR=0.9
    while True:
        bidx=np.argmin(fit); hist.append(float(fit[bidx]))
        for i in range(pop_size):
            idxs=[j for j in range(pop_size) if j!=i]
            a,b,c=np.random.choice(idxs,3,replace=False)
            mutant=X[a]+F*(X[b]-X[c]); mutant=safe_clip(mutant,bounds)
            cross=np.random.rand(D)<CR
            if not np.any(cross): cross[np.random.randint(D)]=True
            trial=np.where(cross, mutant, X[i])
            try: val=f(trial)
            except StopIteration: return X[bidx].copy(), float(fit[bidx]), hist
            if val<fit[i]: X[i]=trial; fit[i]=val


def run_woa(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X]); hist=[]; t=0
    max_iters=max(1,(max_fes-pop_size)//pop_size)
    while True:
        bidx=np.argmin(fit); best=X[bidx].copy(); bfit=fit[bidx]; hist.append(float(bfit))
        a=2-2*(t/max_iters)
        for i in range(pop_size):
            r=np.random.rand(); A=2*a*np.random.rand(D)-a; C=2*np.random.rand(D)
            p=np.random.rand(); l=np.random.uniform(-1,1)
            if p<0.5:
                if np.linalg.norm(A)<1:
                    Xn=best-A*np.abs(C*best-X[i])
                else:
                    rand=X[np.random.randint(pop_size)]
                    Xn=rand-A*np.abs(C*rand-X[i])
            else:
                Xn=np.abs(best-X[i])*np.exp(1*l)*np.cos(2*np.pi*l)+best
            Xn=safe_clip(Xn,bounds)
            try: fit[i]=f(Xn)
            except StopIteration: return best, float(bfit), hist
            X[i]=Xn
        t+=1


def run_ga(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X]); hist=[]
    def tournament():
        a,b=np.random.choice(pop_size,2,replace=False)
        return X[a].copy() if fit[a]<fit[b] else X[b].copy()
    while True:
        bidx=np.argmin(fit); hist.append(float(fit[bidx]))
        newX=[]
        while len(newX)<pop_size:
            p1=tournament(); p2=tournament()
            alpha=np.random.rand(D)
            c1=alpha*p1+(1-alpha)*p2
            c2=alpha*p2+(1-alpha)*p1
            for c in [c1,c2]:
                if np.random.rand()<0.2:
                    j=np.random.randint(D); lb,ub=bounds[j]; c[j]=lb+np.random.rand()*(ub-lb)
                newX.append(safe_clip(c,bounds))
                if len(newX)>=pop_size: break
        X=np.array(newX[:pop_size])
        for i in range(pop_size):
            try: fit[i]=f(X[i])
            except StopIteration: return X[bidx].copy(), float(hist[-1]), hist


def run_ssa(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X]); hist=[]; t=0
    max_iters=max(1,(max_fes-pop_size)//pop_size)
    while True:
        idx=np.argmin(fit); food=X[idx].copy(); best=fit[idx]; hist.append(float(best))
        c1=2*np.exp(-((4*t/max_iters)**2))
        for i in range(pop_size):
            if i < pop_size//2:
                c2=np.random.rand(D); c3=np.random.rand(D)
                step=c1*((np.array([lb for lb,_ in bounds])+c2*(np.array([ub for _,ub in bounds])-np.array([lb for lb,_ in bounds]))) )
                Xn=np.where(c3<0.5, food+step, food-step)
            else:
                Xn=(X[i]+X[i-1])/2.0
            Xn=safe_clip(Xn,bounds)
            try: fit[i]=f(Xn)
            except StopIteration: return food, float(best), hist
            X[i]=Xn
        t+=1


def run_hho(func, bounds, pop_size=50, max_fes=60000, seed=0):
    np.random.seed(seed)
    D=len(bounds); f=ObjectiveCounter(func,max_fes)
    X=_init_pop(bounds,pop_size); fit=np.array([f(x) for x in X]); hist=[]; t=0
    max_iters=max(1,(max_fes-pop_size)//pop_size)
    while True:
        idx=np.argmin(fit); rabbit=X[idx].copy(); best=fit[idx]; hist.append(float(best))
        E1=2*(1 - t/max_iters)
        Xm=np.mean(X, axis=0)
        for i in range(pop_size):
            E0=2*np.random.rand()-1; E=E1*E0; J=2*(1-np.random.rand())
            q=np.random.rand(); r=np.random.rand()
            if abs(E)>=1:
                rand=X[np.random.randint(pop_size)]
                Xn=rand - np.random.rand(D)*np.abs(rand-2*np.random.rand(D)*X[i])
            else:
                if r>=0.5 and abs(E)<0.5:
                    Xn=rabbit - E*np.abs(rabbit-X[i])
                elif r>=0.5 and abs(E)>=0.5:
                    Xn=(rabbit-X[i]) - E*np.abs(J*rabbit-X[i])
                elif r<0.5 and abs(E)>=0.5:
                    Y=rabbit - E*np.abs(J*rabbit-X[i])
                    Xn=Y + np.random.randn(D)*0.01
                else:
                    Y=rabbit - E*np.abs(J*rabbit-Xm)
                    Xn=Y + np.random.randn(D)*0.01
            Xn=safe_clip(Xn,bounds)
            try: fit[i]=f(Xn)
            except StopIteration: return rabbit, float(best), hist
            X[i]=Xn
        t+=1

ALGORITHMS = {
    'MSA': run_msa,
    'IMSA': run_imsa,
    'PSO': run_pso,
    'GWO': run_gwo,
    'DE': run_de,
    'WOA': run_woa,
    'GA': run_ga,
    'SSA': run_ssa,
    'HHO': run_hho,
}
