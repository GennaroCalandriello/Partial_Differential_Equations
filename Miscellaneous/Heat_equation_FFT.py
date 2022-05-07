import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import module.graph as graph
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})
plot_kind='dynamic'

a = 1  # thermal diffusivity constant
L = 100  # length of domain
N = 1000  # Number of discretization points
dx = L/N
x = np.arange(-L/2, L/2, dx)

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial condition
u0 = np.zeros(np.shape(x))
u0[int((L/2-L/10)/dx):int((L/2+L/10)/dx)] = 1
u0hat = np.fft.fft(u0)

# Using scipy we need to recast the state u0hat from an N-element complex
# vector to a 2N-element real vector
u0hat_ri = np.concatenate((u0hat.real, u0hat.imag))

# Simulate in Fourier frequency domain
dt = 0.1
t = np.arange(0, 10, dt)


def rhsHeat(uhat_ri, t,  kappa, a):
    uhat = uhat_ri[:N]+(1j)*uhat_ri[N:]
    d_uhat = -a**2*kappa**2*uhat  # this is the right hand side
    d_uhat_ri = np.concatenate((d_uhat.real, d_uhat.imag)).astype('float64')
    return d_uhat_ri


uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))

uhat = uhat_ri[:, :N]+(1j)*uhat_ri[:, N:]

u = np.zeros(np.shape(uhat))

# inverse Fourier transform
for k in range(len(t)):
    u[k, :] = np.fft.ifft(uhat[k, :])

u = u.real
u_plot = u[0:-1:10, :]

# =============================Dynamic and Static Plot=====================
if plot_kind=='static':
    graph.static_plot(u, x)

if plot_kind=='dynamic':
    graph.animate_plot(L, x, u)
