import logging
import numpy as np
from numba import njit
import module.function as func
import module.graph as graph
import matplotlib.pyplot as plt
import time
import shutil
import os

'''This program executes the integration of the Van der Pol Fokker Planck varying each time the
extension of the lattice and compare the convegence of the solution. The comparison is done through the maximum
value of each array for each time slice, at given N'''

logging.basicConfig(level=logging.INFO)

time_steps = 50000
L = 2*np.pi
INTV = 200
dt = 2e-3
D = 0.5


@njit(fastmath=True, cache=True)
def alternate_direction_implicit(N):
    '''Execute the integration via Alternate Direction Implicit for a NxN lattice extension'''
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters(N)
    cc = 0
    p_new = np.zeros(np.shape(p))
    p_total = np.zeros((round(time_steps/INTV), N, N))

    for t in range(time_steps):
        # BC
        p = func.boundary_conditions('periodic', N, p)
        tt = t*dt

        # implicito su x ed esplicito su y
        p_new = func.implicit_x_explicit_y(
            N, tt, p, p_new, x, y, betax, betay, alphax, alphay)

        # implicito su y esplicito su x
        p = func.implicit_y_explicit_x(
            N, tt, p, p_new, x, y, betax, betay, alphax, alphay)

        if t % INTV == 0:
            p_total[cc] = p
            cc += 1
            print(f'time step: {t}')

    return(p_total)


def max_value_in_a_square(p_total, x, y):
    '''Find the max value in a square of a mesh grid. It is an experiment to verify the convergence of the method without
    an analytical solution'''
    max_list = []
    i_list = []
    j_list = []
    for i in range(N):
        if (x[i] >= 0.8 or x[i] <= 1.):
            i_list.append(i)
    for j in range(N):
        if (y[j] >= 0.8 or y[j] <= 1.):
            j_list.append(j)
    i_max = max(i_list)
    i_min = min(i_list)
    j_max = max(j_list)
    j_min = min(j_list)
    for t in range(len(p_total)):
        max_list.append(np.amax(p_total[t, i_min:i_max, j_min:j_max]))
    return(np.array(max_list))


if __name__ == '__main__':
    execute = True
    plotting = True
    vtk = False #it is necessary ParaView to read this kind of dataset
    static_plot=True

    N_array = [25, 400]

    if execute:
        max_array = np.zeros((len(N_array), round(time_steps/INTV)))
        c = 0
        for N in N_array:
            print(f'Executing for N={N}')
            time.sleep(2)
            x, y, dx, _, _, _, _, _ = func.parameters(N)
            p_total = alternate_direction_implicit(N)
            max_array[c] = max_value_in_a_square(p_total, x, y)
            c += 1
            
            if plotting:
                # graph.animate_matplotlib(x, y, p_total)
                if (vtk==True and N==400):
                    print(f'Writing .vtk file for N = {N}')
                    # path to save file in .vtk format for ParaView
                    path_save = f'OneDrive/Desktop/Github_projects/Van_der_Pol_Fokker_Planck/VdP_ADI/VdP_N_{N}'   #change the path!
                    if os.path.exists(path_save): shutil.rmtree(path_save)
                    os.makedirs(path_save)
                    for w in range(len(p_total)):
                        graph.writeVtk(w, p_total[w], N, dx, path_save)
                
                if (static_plot==True and N==400):
                    for w in range(len(p_total)):
                        if (w*INTV)%300==0: #each time which the static plot is executed and saved
                            graph.static_plot(x, y, p_total[w], w*INTV)
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

