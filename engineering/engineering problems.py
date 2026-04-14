import math
import numpy as np

# 1. Pressure Vessel

def pv_repair(x):
    x=np.array(x,dtype=float).copy()
    x[0]=np.clip(math.ceil(x[0]/0.0625)*0.0625,0.0625,6.1875)
    x[1]=np.clip(math.ceil(x[1]/0.0625)*0.0625,0.0625,6.1875)
    x[2]=np.clip(x[2],10,200); x[3]=np.clip(x[3],10,200)
    x[0]=max(x[0], math.ceil((0.0193*x[2])/0.0625)*0.0625)
    x[1]=max(x[1], math.ceil((0.00954*x[2])/0.0625)*0.0625)
    return x

def pressure_vessel(x):
    x=pv_repair(x); x1,x2,x3,x4=x
    cost=0.6224*x1*x3*x4 + 1.7781*x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
    g1=-x1+0.0193*x3; g2=-x2+0.00954*x3
    g3=1296000 - math.pi*x3**2*x4 - (4/3)*math.pi*x3**3
    g4=x4-240
    pen=sum(1e8*max(0,g)**2 for g in [g1,g2,g3,g4])
    return cost+pen

def pressure_vessel_true(x):
    x=pv_repair(x); x1,x2,x3,x4=x
    cost=0.6224*x1*x3*x4 + 1.7781*x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
    return x, cost

# 2. Tension/Compression Spring

def spring_repair(x):
    x=np.array(x,dtype=float).copy(); x[0]=np.clip(x[0],0.05,2.0); x[1]=np.clip(x[1],0.25,1.3); x[2]=np.clip(x[2],2.0,15.0); return x

def spring(x):
    x=spring_repair(x); d,D,N=x
    cost=(N+2)*D*d**2
    g1=1 - (D**3*N)/(71785*d**4)
    g2=(4*D**2 - d*D)/(12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1
    g3=1 - (140.45*d)/(D**2*N)
    g4=(D+d)/1.5 - 1
    pen=sum(1e8*max(0,g)**2 for g in [g1,g2,g3,g4])
    return cost+pen

def spring_true(x):
    x=spring_repair(x); d,D,N=x
    return x,(N+2)*D*d**2

# 3. Welded Beam

def welded_repair(x):
    x=np.array(x,dtype=float).copy()
    lbs=[0.1,0.1,0.1,0.1]; ubs=[2.0,10.0,10.0,2.0]
    for i in range(4): x[i]=np.clip(x[i], lbs[i], ubs[i])
    return x

def welded_beam(x):
    x=welded_repair(x); h,l,t,b=x
    P=6000; L=14; E=30e6; G=12e6; tau_max=13600; sigma_max=30000
    cost=1.10471*h**2*l + 0.04811*t*b*(14+l)
    M=P*(L+l/2); R=(l**2/4 + ((h+t)/2)**2)**0.5
    J=2*(2**0.5*h*l*(l**2/12 + ((h+t)/2)**2))
    tau1=P/(2**0.5*h*l)
    tau2=M*R/J
    tau=(tau1**2 + 2*tau1*tau2*l/(2*R) + tau2**2)**0.5
    sigma=6*P*L/(b*t**2)
    delta=4*P*L**3/(E*b*t**3)
    Pc=(4.013*E/(6*L**2))*t*b**3*(1 - t/(2*L)*(E/(4*G))**0.5)
    g=[tau-tau_max, sigma-sigma_max, h-b, 0.125-h, delta-0.25, P-Pc]
    pen=sum(1e8*max(0,gi)**2 for gi in g)
    return cost+pen

def welded_true(x):
    x=welded_repair(x); h,l,t,b=x
    return x, 1.10471*h**2*l + 0.04811*t*b*(14+l)

# 4. Gear Train

def gear_repair(x):
    x=np.array(x,dtype=float).copy()
    for i in range(4): x[i]=np.clip(round(x[i]),12,60)
    return x

def gear_train(x):
    x=gear_repair(x); x1,x2,x3,x4=x
    return (1/6.931 - (x1*x2)/(x3*x4))**2

def gear_true(x):
    x=gear_repair(x)
    return x, (1/6.931 - (x[0]*x[1])/(x[2]*x[3]))**2

PROBLEMS = {
    'pressure_vessel': {'func': pressure_vessel, 'true': pressure_vessel_true, 'bounds': [(0.0625,6.1875),(0.0625,6.1875),(10,200),(10,200)]},
    'spring': {'func': spring, 'true': spring_true, 'bounds': [(0.05,2.0),(0.25,1.3),(2.0,15.0)]},
    'welded_beam': {'func': welded_beam, 'true': welded_true, 'bounds': [(0.1,2.0),(0.1,10.0),(0.1,10.0),(0.1,2.0)]},
    'gear_train': {'func': gear_train, 'true': gear_true, 'bounds': [(12,60),(12,60),(12,60),(12,60)]},
}
