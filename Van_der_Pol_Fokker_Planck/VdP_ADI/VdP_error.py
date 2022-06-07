import os
import shutil
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import module.function as func
from scipy.optimize import curve_fit
import random


"""This program performs iteratively (not in parallel) the error calculus changing dt from 2e-1 to 2e-3.
After saving a certain number of time slices it calculates the error by choosing, for each (Nt, dt), the max of 
a set of errors calculated for each time slice. Or, if err_calc_max==True, it performs the sigma calculus only
on the last time slices for each dt, when the system i completely evoluted to the equilibrium """

N = 80  # space extension
Nt_max = 100000
Nt_min = 10000
N_array = [30, 50, 80, 100]


@njit(fastmath=True, cache=True)
def alternate_direction_implicit(N, dt):
    """Execute the integration via Alternate Direction Implicit for a NxN lattice extension"""
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters_var_dt(N, dt)
    time_steps = round(200 / dt)
    p_new = np.zeros(np.shape(p))
    p_total = np.zeros((time_steps, N, N))
    print("Time steps", time_steps)

    for t in range(time_steps):
        if t % (1000) == 0:
            print(f"Exe for Nt: {time_steps}, step: {t}")
        # BC
        p = func.boundary_conditions("absorbing", N, p)
        tt = t * dt

        # implicito su x ed esplicito su y
        p_new = func.implicit_x_explicit_y(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay
        )

        # implicito su y esplicito su x
        p = func.implicit_y_explicit_x(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay
        )

        p_total[t] = p

    return p_total


def f_fit(x, a, b):
    return a * x ** 3


if __name__ == "__main__":

    error_calc = True  # execute the calculus on various t_slices for each Nt and take the maximum value at various step during the evolution
    error_calc_max = True  # execute the calculus for only the last t_slice for each Nt, once the system is completely evoluted
    integral_error = False

    num_of_dt = 20  # number of various dt that you want
    t_slices = np.array(
        [2 ** n for n in range(8)]
    )  # time slices that we save for each Nt
    Nt_array = np.linspace(Nt_min, Nt_max, num_of_dt).astype("int32")
    dt_array = (
        200 / Nt_array
    )  # we maintain constant the max time extension for the evolution of the system
    execute = "n"  # input('Do you want to execute the program and overwrite the data? (y/n) ')

    if execute == "y":
        # creating paths
        print("Execution error simulations for N =", N)
        path0 = f"data{N}"
        if os.path.exists(path0):
            shutil.rmtree(path0)
        os.makedirs(path0)

        for Nt in Nt_array:
            os.makedirs(f"{path0}/u_Nt_{Nt}")

        for t in range(len(dt_array)):
            u_tot = alternate_direction_implicit(N, dt_array[t])
            tsave_arr = (Nt_array[t] / t_slices).astype("int32")
            path_u = os.path.join(path0, f"u_Nt_{Nt_array[t]}")
            c = 0

            for tsave in tsave_arr:
                np.savetxt(f"{path_u}/u_{c}.txt", u_tot[tsave - 1])
                c += 1

    if (
        error_calc
    ):  # calculus of the error at various t_slices for each Nt, and then take the maximum value of the errors
        # u0 for Nt=60000, dt=2e-3
        errore_prec_succ = (
            False  # per ottenere l'errore tra il dt precedente e il successivo
        )
        ts = len(t_slices)  # number of time slices we consider

        max_value = []
        p = 0
        prec = np.array([i for i in range(0, 19)])
        succ = np.array([i for i in range(1, 20)])
        err_for_all_N = np.zeros((len(N_array), len(Nt_array)))

        c = 0
        for N in N_array:
            err = []
            for p in range(len(Nt_array)):
                path_u = f"data{N}/u_Nt_{Nt_array[p]}"
                path_u0 = f"data{N}/u_Nt_{Nt_max}"
                # path_u_succ = f"data{N}/u_Nt_{Nt_array[succ[p]]}"
                u_Nt = np.zeros((ts, N, N))
                u0 = u_Nt.copy()
                u_Nt_succ = u_Nt.copy()

                for i in range(ts):

                    u_Nt[i] = np.loadtxt(f"{path_u}/u_{i}.txt")
                    u0[i] = np.loadtxt(f"{path_u0}/u_{i}.txt")
                    # u_Nt_succ[i] = np.loadtxt(f"{path_u_succ}/u_{i}.txt")

                err_n = []
                if errore_prec_succ:

                    for i in range(ts):
                        err_n.append(np.amax(np.abs(u_Nt[i] - u_Nt_succ[i])))
                    err.append(max(err_n))

                else:

                    for i in range(ts):
                        err_n.append(np.amax(np.abs(u_Nt[i] - u0[i])))

                    err.append(max(err_n))
            err_for_all_N[c, :] = np.array(err)
            c += 1

        c = 0

        # PLOT
        for N in N_array:

            err_N = err_for_all_N[c]
            rgb = (random.random(), random.random(), random.random())
            plt.scatter(
                dt_array[:-1] - min(dt_array),
                err_for_all_N[c, 1:],
                s=20,
                c=[rgb],
                label=f"N = {N}",
            )
            c += 1
            err_N = 0
        plt.title("Andamento errore per vari dt", fontsize=25)
        plt.xlabel(r"d$\tau$", fontsize=15)
        plt.ylabel(r"$\sigma$(dt)", fontsize=15)
        plt.legend()
        # plt.yscale("log")
        plt.show()

        # curve fitting differences between the value of error and the previous one in err array
        err = np.array(err)
        err_diff = []

        for k in range(len(err)):
            err_diff.append(np.abs((err[k] - err[k - 1])))

        opt, _ = curve_fit(f_fit, dt_array[1:], err_diff[1:], maxfev=100000)
        x = dt_array[1:-1]
        y = f_fit(dt_array[1:-1], a=opt[0], b=opt[1])

        plt.plot(x, y, "--", c="green", label=r"$ax^2$")
        plt.scatter(
            dt_array[1:], err_diff[1:], label="$\sigma_i-\sigma_{i-1}$", s=5, c="blue"
        )
        plt.xlabel("dt", fontsize=15)
        plt.ylabel(r"$\Delta \sigma$", fontsize=15)
        plt.title(r"Andamento $\Delta \sigma$", fontsize=25)
        plt.legend()
        plt.show()
        print(f"a={opt[0]}, b={opt[1]}")

    if error_calc_max:  # calculus of error only on the last slice for each Nt, dt

        err_for_all_N = np.zeros((len(N_array), len(Nt_array)))
        c = 0
        for N in N_array:
            error = []
            path_u0 = f"data{N}/u_Nt_{Nt_max}"
            u_tot = np.zeros((len(Nt_array), N, N))
            for t in range(len(Nt_array)):
                u_tot[t] = np.loadtxt(
                    f"data{N}/u_Nt_{Nt_array[t]}/u_{len(t_slices)-1}.txt"
                )
                u0 = np.loadtxt(f"{path_u0}/u_{7}.txt")

                error.append(np.amax(np.abs(u_tot[t] - u0)))
            err_for_all_N[c] = np.array(error)
            c += 1

        c = 0
        # PLOT
        for N in N_array:
            rgb = (random.random(), random.random(), random.random())
            plt.scatter(
                dt_array[:-1] - min(dt_array),
                err_for_all_N[c, 1:],
                s=20,
                c=[rgb],
                label=f"N = {N}",
            )
            c += 1
        plt.title("Andamento errore per vari dt, ultima slice", fontsize=25)
        plt.xlabel(r"d$\tau$", fontsize=15)
        plt.ylabel(r"$\sigma$(dt)", fontsize=15)
        # plt.yscale("log")
        plt.legend()
        plt.show()

    # if integral_error:  ####rivedere

    #     u_tot = np.zeros((len(Nt_array), N, N))
    #     u0 = np.loadtxt(f"data/u_Nt_{Nt_max}/u_{len(t_slices)-1}.txt")
    #     error = []
    #     error_integral = []
    #     x, y, _, _, _, _, _, _ = func.parameters(N)
    #     for t in range(len(Nt_array)):
    #         u_tot[t] = np.loadtxt(f"data/u_Nt_{Nt_array[t]}/u_{len(t_slices)-1}.txt")
    #         value = np.zeros((N, N))
    #         for i in range(1, N - 1):
    #             for j in range(1, N - 1):
    #                 value[i, j] = np.abs(u_tot[t, i, j] - u_tot[t - 1, i, j])
    #         error_integral.append(
    #             simps(simps(value, y), x)
    #         )  # evaluate the integral of the 2D array

    #     print(error_integral)
    #     print("max u0", np.amax(u0))
    #     plt.plot(dt_array[1:], error_integral[1:], "o", c="red")
    #     plt.title("Integral error on the last slice for each dt", fontsize=20)
    #     plt.xlabel("dt", fontsize=15)
    #     plt.ylabel(r"$\sigma$(dt)", fontsize=15)
    #     plt.show()

