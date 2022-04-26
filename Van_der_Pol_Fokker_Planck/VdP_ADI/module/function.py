import numpy as np
from numba import njit, float64, int32
import matplotlib.pyplot as plt

L = 2*np.pi
dt = 2e-3
D = 0.5
cos=True #parameter that regulates the presence of a periodic drift

@njit(fastmath=True, cache=True)
def initial_state_(N, p, x, y, L):
    '''Construct the initial state'''
    a = 1
    for i in range(N):
        for j in range(N):
            p[i, j] = np.exp(-((x[i]-0.3*L)**2+(y[j]-0.3*L)
                             ** 2)/(2*a))  # shift di 0.3*L
    return p

@njit(fastmath=True, cache=True)
def parameters(N):
    '''Construct parameters and initial condition's array for each N'''
    x = np.linspace(-L, L, N)
    y = x
    p = np.zeros((N, N))
    dx = (x[1]-x[0])
    dy = dx
    betax = 0.5*dt/dx
    betay = 0.5*dt/dy
    alphax = dt*D/dx**2
    alphay = dt*D/dy**2
    p = initial_state_(N, p, x, y, L)
    return (x, y, dx, p, betax, betay, alphax, alphay)

@njit()
def g(x, y):
    '''Drift func on the x first derivative for Van der Pol FP'''
    return (y)

@njit()
def f(x, y, t):
    '''Drift func on the y first derivative for Van der Pol FP'''
    w = 0.3
    a0 = 0.1
    a = a0*(np.cos(w*t))
    if cos: return (x-a*y*(1-x**2))
    else: return (x-a0*y*(1-x**2))
    
@njit()
def boundary_conditions(kind, N, matrix):
    '''Impose boundary conditions; kind should be: absorbing, periodic or reflecting'''
    if kind == 'absorbing':
        matrix[0, :] = 0
        matrix[:, 0] = 0
        matrix[N-1, :] = 0
        matrix[:, N-1] = 0
    if kind == 'periodic':
        matrix[0, :] = matrix[N-2, :]
        matrix[N-1, :] = matrix[1, :]
        matrix[:, 0] = matrix[:, N-2]
        matrix[:, N-1] = matrix[:, 1]
    if kind == 'reflecting':
        matrix[0, :] = matrix[1, :]
        matrix[:, 0] = matrix[:, 1]
        matrix[N-2, :] = matrix[N-1, :]
        matrix[:, N-2] = matrix[:, N-1]
    return matrix

@njit(float64[:](int32, float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True) #questa funzione risolve il sistema di equazioni invece di invertire matrici
def solve_matrix(n, lower_diagonal, main_diagonal, upper_diagonal, solution_vector):

    '''Solve systems of equations through Thomas Algorithm instead of inverting matrices. It returns
       the same solution of np.linalg.solve'''

    w=np.zeros(n-1)
    g=np.zeros(n)
    result=np.zeros(n)

    w[0]=upper_diagonal[0]/main_diagonal[0]
    g[0]=solution_vector[0]/main_diagonal[0]

    for i in range(1, n-1):
        w[i]=upper_diagonal[i]/(main_diagonal[i]-lower_diagonal[i-1]*w[i-1])
    for i in range(1, n):
        g[i]=(solution_vector[i]-lower_diagonal[i-1]*g[i-1])/(main_diagonal[i]-lower_diagonal[i-1]*w[i-1])
    result[n-1]=g[n-1]
    for i in range(n-1, 0, -1):
        result[i-1]=g[i-1]-w[i-1]*result[i]
    return result #restituisce la stessa soluzione che con linalg.solve

@njit(fastmath=True, cache=True)
def implicit_x_explicit_y(N, tt, p: np.array, p_new: np.array,  x: np.array, y: np.array, betax, betay, alphax, alphay):
    '''Execute the ADI implicitly on x and explicitly on y'''
    main = np.ones(N)
    up_diag = np.ones(N)
    low_diag = np.ones(N)
    step = np.zeros(N)
    temp = np.zeros(N)
    for j in range(1, N-1):
        for i in range(N):
            up_diag[i] = 0.5*betax*g(x[i+1], y[j])
            low_diag[i] = -0.5*betax*g(x[i-1], y[j])
        for i in range(N):
            step[i] = p[i, j]+0.5*betay*(f(x[i], y[j+1], tt)*p[i, j+1]-f(x[i], y[j-1], tt)*p[i, j-1]) +\
                alphay*0.5*(p[i, j+1]-2*p[i, j]+p[i, j-1])
        temp = solve_matrix(
            N, low_diag[1:], main, up_diag[:N-1], step)
        for i in range(N):
            p_new[i, j] = temp[i]
    return p_new

@njit(fastmath=True, cache=True)
def implicit_y_explicit_x(N, tt, p: np.array, p_new: np.array, x: np.array, y: np.array, betax, betay, alphax, alphay):
    '''Execute the ADI implicitly on y and explicitly on x'''
    main = np.ones(N)*(1+alphay)
    up_diag = np.ones(N)
    low_diag = np.ones(N)
    step = np.zeros(N)
    temp = np.zeros(N)
    for i in range(1, N-1):
        for j in range(N):
            up_diag[j] = -0.5*betay*f(x[i], y[j+1], tt+dt/2)-0.5*alphay
            low_diag[j] = 0.5*betay*f(x[i], y[j-1], tt+dt/2)-0.5*alphay
        for j in range(N):
            step[j] = p_new[i, j]-0.5*betax * \
                (g(x[i+1], y[j])*p[i+1, j]-g(x[i-1], y[j])*p[i-1, j])
        temp = solve_matrix(
            N, low_diag[1:], main, up_diag[:N-1], step)
        for j in range(N):
            p[i, j] = temp[j]
    return p
