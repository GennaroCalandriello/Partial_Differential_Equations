import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64, int32
import module.function as func
import module.graph as graph
import time
import shutil
import os

"""!!!Warning!!!! This explicit scheme is not accurate like ADI"""

"""This program integrates the Fokker Planck of a Van der Pol by explicit O(dt^2) scheme. We reccomend to use ADI that is more accurate"""

time_steps = 50000
dt = 2e-4
INTV = 500


@njit(float64[:, :, :](int32), fastmath=True, cache=True)
def calcolo(N):

    p_total = np.zeros((round(time_steps / INTV), N, N))
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters(N, dt)
    p_new = p
    cc = 0

    for t in range(time_steps):
        tt = dt * t
        for i in range(N - 1):
            for j in range(N - 1):
                p_new[i, j] = (
                    p[i, j]
                    - betax
                    * (
                        func.g(x[i + 1], y[j]) * p[i + 1, j]
                        - func.g(x[i - 1], y[j]) * p[i - 1, j]
                    )
                    + betay
                    * (
                        func.f(x[i], y[j + 1], tt) * p[i, j + 1]
                        - func.f(x[i], y[j - 1], tt) * p[i, j - 1]
                    )
                    + alphay * (p[i, j + 1] - 2 * p[i, j] + p[i, j - 1])
                )

        p_new = func.boundary_conditions("absorbing", N, p_new)
        p = p_new

        if t % INTV == 0:
            print(f" Time step: {t} for N: {N}")
            p_total[cc] = p_new
            cc += 1

    return p_total


if __name__ == "__main__":
    execute = True
    vtk = False
    plot = True
    static_plot = False
    N_array = [150]
    if execute == True:
        _, _, _, _, _, _, _, alphay = func.parameters(N_array[0])
        print(f"alphay = {alphay}")
        time.sleep(2)
        c = 0
        max_array = np.zeros((len(N_array), round(time_steps / INTV)))

        for N in N_array:
            print(f"Execution for N: {N}")
            time.sleep(2)
            x, y, dx, _, _, _, _, _ = func.parameters(N)
            p_total = calcolo(N)
            c += 1

            if static_plot == True and N == max(N_array):
                print(f"Static plot saving for: {max(N_array)}")
                for sp in range(len(p_total)):
                    if (sp * INTV) % 5000 == 0:
                        graph.static_plot(x, y, p_total[sp], sp * INTV)

            if plot:
                graph.animate_matplotlib(x, y, p_total)

            if vtk == True and N == max(N_array):
                print(f"Writing .vtk for N: {N}")
                path_vtk = f"OneDrive/Desktop/Github_projects/Van_der_Pol_Fokker_Planck/VdP_explicit/plots/VdP_N_{N}"
                if os.path.exists(path_vtk):
                    shutil.rmtree
                os.makedirs(path_vtk)
                for t in range(len(p_total)):
                    graph.writeVtk(t, p_total[t], N, dx, path_vtk)

