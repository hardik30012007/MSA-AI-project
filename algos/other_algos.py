import numpy as np
from bench_utils import ObjectiveCounter, safe_clip

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
    'PSO': run_pso,
    'GWO': run_gwo,
    'DE': run_de,
    'WOA': run_woa,
    'GA': run_ga,
    'SSA': run_ssa,
    'HHO': run_hho,
}