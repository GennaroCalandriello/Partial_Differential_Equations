import numpy as np
from numpy.fft import rfft, irfft
from numpy import pi, cos
import module.graph as graph

'''This program integrates the Korteweg-de Vries 1D nonlinear partial differential equation through Fast Fourier Transform:
KdV equation: u_t=-u*u_x-delta*u_xxx
The PDE is converted, through Fourier analysis, in a ODE and integrated at each time step by a 4-th order Runge-Kutta method '''

graphic='static'
delta=0.022**2
TB=1./np.pi

N=500
dt=1e-5*TB
INTV=1000
time_steps=int(0.2e6)
L=2*np.pi
#Initial condition
x=np.linspace(-L, L , N)
u=cos(pi*x)
v=rfft(u)

k=pi*np.arange(len(v))
k2=k*k
k3=k*k*k
v_final=np.zeros((round(time_steps/INTV), len(x))) #final 2D array containing each time step s.t. time_steps%INTV=0

# @njit()
def calculus(v):
    '''Execute the calculus of the right hand side of the KdV equation for each time step'''
    u=irfft(v)
    u_x=irfft(1j*k*v)
    conv=-rfft(u*u_x)
    disp=+delta*1j*k3*v
    dv=conv+disp
    return dv

#Time loop, the vector v is updated at each time step
cc=0
for nt in range(time_steps):
    t=dt/TB*nt

    #4-th order Runge-Kutta method
    dv0=calculus(v)
    v1=v+0.5*dt*dv0
    dv1=calculus(v1)
    v2=v+0.5*dt*dv1
    dv2=calculus(v2)
    v3=v+dt*dv2
    dv3=calculus(v3)

    v=v+dt/6.*(dv0+2.*dv1+2.*dv2+dv3)

    if nt%INTV==0:
        print(f'time step {nt}')
        u=irfft(v)
        v_final[cc]=u
        cc+=1

#graphic part
if graphic=='animate':   
    graph.animate_plot(L, x, v_final)
if graphic=='static':
    graph.static_plot(v_final, x)




