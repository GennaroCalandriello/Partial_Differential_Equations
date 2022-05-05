from importlib.resources import path
import os
import shutil
import numpy as np
from numba import njit
import module.function as func
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

'''!!!WARNING!!! ---> this program should be runned on a supercomputer due to parallelization. 
Execution on a normal pc or laptop can degenerate in a CRASH. If we want to try, you should regulate
the spatial extension at max N=50 for a max of 5, 6 dt (so 5 or 6 processes in parallel). This is due to the 
enormous array that the multiprocessing generates (if N=100 and we want to analyze the system for Nt=6000, the smallest
one, we have a 3D array with 6000*100*100, if total Nts are 5, we have 5*Nt*100*100 with Nt in (6000, 60000)!!!

Remember: you're the only responsible for your CRASH!!!

This program performs in parallel the error calculus changing dt from 2e-1 to 2e-3.
After saving a certain number of time slices it calculates the error by choosing, for each (Nt, dt), the max of 
a set of errors calculated for each time slice. Or, if err_calc_max==True, it performs the sigma calculus only
on the last time slices for each dt, when the system i completely evoluted to the equilibrium '''

@njit(fastmath=True, cache=True)
def alternate_direction_implicit(N, dt):
    '''Execute the integration via Alternate Direction Implicit for a NxN lattice extension'''
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters_var_dt(N, dt)
    time_steps=round(120/dt)
    p_new = np.zeros(np.shape(p))
    p_total = np.zeros((time_steps, N, N))
    print('time steps', time_steps)

    for t in range(time_steps):
        if t%(time_steps/5)==0: print(f'Exe for Nt: {time_steps}, step: {t}' )
        # BC
        p = func.boundary_conditions('absorbing', N, p)
        tt = t*dt

        # implicito su x ed esplicito su y
        p_new = func.implicit_x_explicit_y(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)

        # implicito su y esplicito su x
        p = func.implicit_y_explicit_x(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)

        p_total[t]=p

    return(p_total)


if __name__ == '__main__':
    execute = True
    error_calc=False #execute the calculus on various t_slices for each Nt and take the maximum value at various step during the evolution
    error_calc_max=True #execute the calculus for only the last t_slice for each Nt, once the system is completely evoluted
    N=50

    tt_max=120 #max time extension 
    num_of_dt=10
    t_slices=np.array([2**n for n in range(8)]) #t that we save for each Nt
    Nt_array=np.linspace(6000, 60000, num_of_dt).astype('int32')
    print('Array of Nt: ', Nt_array)
    dt_array=120/Nt_array #we maintain constant the max time extension for the evolution of the system

    if execute:
        #creating paths
        path0='data'
        shutil.rmtree(path0)
        os.makedirs(path0)
        for Nt in Nt_array:
            os.makedirs(f'{path0}/u_Nt_{Nt}')

        #parallelizing the execution of ADI for all dt in dt_array
        with multiprocessing.Pool(processes=len(dt_array)) as pool:
            part=partial(alternate_direction_implicit, N)
            results = np.array(pool.map(part, dt_array), dtype='object')
            pool.close()
            pool.join()
        #saving:
        c=0
        for Nt in Nt_array: #selecting various dt (so various Nt)
            tsave_arr=(Nt/t_slices).astype('int32') #define the time slice that should be saved
            tsave_arr.sort() #reordering in crescentini way
            u_tot=results[c]
            print(u_tot)
            d=0
            path=os.path.join(path0,f'u_Nt_{Nt}')
            for tsave in tsave_arr:
                np.savetxt(f'{path}/u_{d}.txt', u_tot[tsave-1])
                d+=1
            c+=1
    
    if error_calc: #calculus of the error at various t_slices for each Nt, and then take the maximum value of the errors
        #u0 for Nt=60000, dt=2e-3
        ts=len(t_slices) #number of time slices we consider
        u0=np.zeros((ts, N, N))
        path_u0='data/u_Nt_60000'
        for i in range(ts):
            u0[i]=np.loadtxt(f'{path_u0}/u_{i}.txt')
        err=[]
        for Nt in Nt_array:
            path_u=f'data/u_Nt_{Nt}'
            u_Nt=np.zeros((ts, N, N))
            #loading data from each Nt, dt
            for i in range(ts):
                u_Nt[i]=np.loadtxt(f'{path_u}/u_{i}.txt')
            #calculus of error by max of max of errors for ts slices:
            err_n=[]
            for i in range(ts):
                err_n.append(np.amax(np.abs(u_Nt[i]-u0[i])))
            err.append(max(err_n))
        
        plt.plot(dt_array, err, 'o', c='blue')
        plt.title('Cazz di errori', fontsize=20)
        plt.xlabel('dt', fontsize=15)
        plt.ylabel(r'$\sigma$(dt)', fontsize=15)
        plt.show()

    if error_calc_max:
        u_tot=np.zeros((len(Nt_array), N, N))
        u0=np.loadtxt(f'data/u_Nt_60000/u_{len(t_slices)-1}.txt')
        error=[]
        for t in range(len(Nt_array)):
            u_tot[t]=np.loadtxt(f'data/u_Nt_{Nt_array[t]}/u_{len(t_slices)-1}.txt')
            error.append(np.amax(np.abs(u_tot[t]-u0)))
        
        plt.plot(dt_array, error, '--o', c='red')
        plt.title('Error on the last slice difference', fontsize=20)
        plt.xlabel('dt', fontsize=15)
        plt.ylabel(r'$\sigma$(dt)', fontsize=15)
        plt.show()

        

        
            






        
        
    




                

                 
            
            

        
    
    