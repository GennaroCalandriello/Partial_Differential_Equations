import os
import shutil
import numpy as np
from numba import njit, float64, int32
import matplotlib.pyplot as plt
from scipy.integrate import simps
import module.function as func
import module.graph as graph
from scipy.optimize import curve_fit


"""This program performs iteratively (not in parallel) the error calculus changing dt from 2e-1 to 2e-3.
After saving a certain number of time slices it calculates the error by choosing, for each (Nt, dt), the max of 
a set of errors calculated for each time slice. Or, if err_calc_max==True, it performs the sigma calculus only
on the last time slices for each dt, when the system i completely evoluted to the equilibrium """

N = 50  # space extension
Nt_max = 1000000
Nt_min = 100000


@njit()
def calcolo(N, dt):
    time_steps = round(200 / dt)
    p_total = np.zeros((time_steps, N, N))
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters(N, dt)
    p_new = p
    cc = 0

    for t in range(time_steps):
        if t % 10000 == 0:
            print("Step", t, "for dt = ", dt)
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

    return p_new


def f_fit(x, a, b):
    return a * x ** 2 + b


if __name__ == "__main__":

    error_calc = True  # execute the calculus on various t_slices for each Nt and take the maximum value at various step during the evolution
    error_calc_max = False  # execute the calculus for only the last t_slice for each Nt, once the system is completely evoluted
    integral_error = False

    num_of_dt = 20  # number of various dt that you want
    t_slices = np.array(
        [2 ** n for n in range(8)]
    )  # time slices that we save for each Nt
    Nt_array = np.linspace(Nt_min, Nt_max, num_of_dt).astype("int32")
    dt_array = (
        200 / Nt_array
    )  # we maintain constant the max time extension for the evolution of the system
    execute = "n"
    print(dt_array)

    if execute == "y":
        # creating paths
        path0 = "data"
        if os.path.exists(path0):
            shutil.rmtree(path0)
        os.makedirs(path0)

        for Nt in Nt_array:
            os.makedirs(f"{path0}/u_Nt_{Nt}")

        for t in range(len(dt_array)):
            u = calcolo(N, dt_array[t])
            path_u = os.path.join(path0, f"u_Nt_{Nt_array[t]}")
            np.savetxt(f"{path_u}/u_0.txt", u)

    if (
        error_calc
    ):  # calculus of the error at various t_slices for each Nt, and then take the maximum value of the errors
        # u0 for Nt=60000, dt=2e-3
        errore_prec_succ = (
            True  # per ottenere l'errore tra il dt precedente e il successivo
        )
        path_u0 = f"data/u_Nt_{Nt_max}"
        u0 = np.loadtxt(f"{path_u0}/u_0.txt")

        err = []
        max_value = []
        p = 0
        prec = np.array([i for i in range(0, 19)])
        succ = np.array([i for i in range(1, 20)])

        for p in range(len(Nt_array) - 1):
            path_u = f"data/u_Nt_{Nt_array[prec[p]]}"
            path_u_succ = f"data/u_Nt_{Nt_array[succ[p]]}"
            u_Nt = np.zeros((N, N))
            u_Nt_succ = u_Nt.copy()
            u_Nt = np.loadtxt(f"{path_u}/u_0.txt")
            u_Nt_succ = np.loadtxt(f"{path_u_succ}/u_0.txt")

            if errore_prec_succ:
                err.append(np.amax(np.abs(u_Nt - u_Nt_succ)))
            else:
                err.append(np.amax(np.abs(u_Nt - u0)))

        print("list of errors: ", err)
        plt.plot(dt_array[:-1], err, "o", c="blue", label=f"N = {N}")
        plt.title("Andamento errore per vari dt", fontsize=20)
        plt.xlabel("dt", fontsize=15)
        plt.ylabel(r"$\sigma$(dt)", fontsize=15)
        plt.legend()
        plt.show()

        # curve fitting differences between the value of error and the previous one in err array
        err = np.array(err)
        err_diff = []

        for k in range(len(err)):
            err_diff.append(np.abs((err[k] - err[k - 1])))

        opt, _ = curve_fit(f_fit, dt_array[1:-1], err_diff[1:], maxfev=100000)
        x = dt_array[1:-1]
        y = f_fit(dt_array[1:-1], a=opt[0], b=opt[1])
        plt.plot(x, y, "--", c="green")
        plt.scatter(dt_array[1:-1], err_diff[1:])
        plt.show()
        print(f"a={opt[0]}, b={opt[1]}")
