import logging
import numpy as np
from numba import njit
import module.function as func
import module.graph as graph
import shutil
import os
import time

time_steps=20000
L=2*np.pi
D=0.5
INTV=80
dt=2e-3

'''This program executes the integration of the FP of a Duffing oscillator through ADI method. The structure of program and functions in module directory
are completely analogue to Van der Pol, except for the structure of discretization. The integration scheme is the same, Alternate Direction Implicit. 
Theoretical details in Duffing_theory.ipynb'''


@njit()
def alternate_direction_implicit(N):
    #parameters and vector's initializations
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters(N)
    cc = 0
    p_new = np.zeros(np.shape(p))
    p_total = np.zeros((round(time_steps/INTV), N, N))

    for t in range(time_steps):
        # BC
        p = func.boundary_conditions('absorbing', N, p)
        tt = t*dt

        # implicito su x ed esplicito su y
        p_new = func.implicit_x_explicit_y(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)

        # implicito su y esplicito su x
        p = func.implicit_y_explicit_x(
           N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)

        if t % INTV == 0:
            p_total[cc] = p
            cc += 1
            print(f'time step: {t}')

    return p_total


if __name__ == '__main__':
    execute = True
    plotting = True
    vtk = True #it is necessary ParaView to read this kind of dataset
    static_plot=False

    N_array = [200] #array of various space extension for the meshgrid

    if execute:
        c = 0
        for N in N_array:
            print(f'Executing for N={N}')
            time.sleep(2)
            x, y, dx, _, _, _, _, _ = func.parameters(N)
            p_total = alternate_direction_implicit(N)
            c += 1
            
            if (plotting==True and N==max(N_array)): #animate plot via matplotlib
                graph.animate_matplotlib(x, y, p_total)

            if (vtk==True and N==max(N_array)): #data for animate plot via ParaView
                print(f'Writing .vtk file for N = {N}')
                # path to save file in .vtk format for ParaView
                path_save = f'Duffing_N_{N}'   #change the path!
                if os.path.exists(path_save): shutil.rmtree(path_save)
                os.makedirs(path_save)
                for w in range(len(p_total)):
                    graph.writeVtk(w, p_total[w], N, dx, path_save)
            
            if (static_plot==True and N==max(N_array)): #static plot via matplotlib
                for w in range(len(p_total)):
                    if ((w*INTV)%10==0 and w*INTV<=300): #each time which the static plot is executed and saved
                        graph.static_plot(x, y, p_total[w], w*INTV)