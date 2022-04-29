import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64, int32
import module.function as func
import module.graph as graph
import time
import shutil
import os

time_steps=50000
dt=2e-3
D=0.5
INTV=200


@njit(float64[:,:,:](int32), fastmath=True, cache=True)
def calcolo(N):

    p_total = np.zeros((round(time_steps/INTV), N, N))
    x, y, dx, p, betax, betay, alphax, alphay=func.parameters(N)
    p_new=p
    cc=0

    for t in range(time_steps):
        tt=dt*t
        for i in range(N-1):
            for j in range(N-1):
                p_new[i, j] = p[i, j]-betax*(func.g(x[i+1], y[j])*p[i+1, j]-func.g(x[i-1], y[j])*p[i-1, j]) +\
                    betay*(func.f(x[i], y[j+1], tt)*p[i, j+1]-func.f(x[i], y[j-1], tt)*p[i, j-1]) +\
                    alphay*(p[i, j+1]-2*p[i, j]+p[i, j-1])

        p_new=func.boundary_conditions('absorbing', N, p_new)
        p=p_new

        if t%INTV==0:
            print(f' Time step: {t} for N: {N}')
            p_total[cc]=p_new
            cc+=1

    return p_total

def max_value_in_a_square(p_total, x, y):
    '''Find the max value in a square of a mesh grid. It is an experiment to verify the convergence of the method without
    an analytical solution'''
    max_list = []
    i_list = []
    j_list = []
    for i in range(N):
        if (x[i] >= 0.6 or x[i] <= 1.):
            i_list.append(i)
    for j in range(N):
        if (y[j] >= 0.6 or y[j] <= 1.):
            j_list.append(j)
    i_max = max(i_list)
    i_min = min(i_list)
    j_max = max(j_list)
    j_min = min(j_list)
    for t in range(len(p_total)):
        max_list.append(np.amax(p_total[t, i_min:i_max, j_min:j_max]))
    return(np.array(max_list))

if __name__=="__main__":
    execute=True
    vtk=False
    plot=False
    static_plot=False
    N_array=[25, 50, 100, 150, 170, 200]
    if execute == True:
        c=0
        max_array = np.zeros((len(N_array), round(time_steps/INTV)))

        for N in N_array:
            print(f'Execution for N: {N}')
            time.sleep(2)
            x, y, dx, _, _, _, _, _=func.parameters(N)
            p_total=calcolo(N)
            max_array[c] = max_value_in_a_square(p_total, x, y)
            c+=1

            if (static_plot==True and N==max(N_array)):
                print(f'Static plot saving for: {max(N_array)}')
                for sp in range(len(p_total)):
                    if (sp*INTV)%5000==0:
                        graph.static_plot(x, y , p_total[sp], sp*INTV)

            if plot:
                graph.animate_matplotlib(x, y, p_total)

            if (vtk==True and N==max(N_array)):
                print(f'Writing .vtk for N: {N}')
                path_vtk=f'OneDrive/Desktop/Github_projects/Van_der_Pol_Fokker_Planck/VdP_explicit/plots/VdP_N_{N}'
                if os.path.exists(path_vtk): shutil.rmtree
                os.makedirs(path_vtk)
                for t in range(len(p_total)):
                    graph.writeVtk(t, p_total[t], N, dx, path_vtk)

            np.savetxt(f'max_try.txt', max_array)
        
        max_arr = np.loadtxt(f'max_try.txt')

        for m in range(len(max_arr)):
            x_arr=np.array(range(round(time_steps/INTV)))
            x_arr=x_arr*INTV
            plt.plot(x_arr,
                    max_arr[m], label=f'N = {N_array[m]}')
            plt.xlabel('t', fontsize=12)
            plt.ylabel(r'$u_{max}$', fontsize=12)
            plt.title(r'Convergenza per $\Delta_x, \Delta_y \to 0 $')
            plt.legend()
        plt.show()


