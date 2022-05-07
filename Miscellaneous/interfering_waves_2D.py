import os
import module.graph as graph
import numpy as np
import logging
from numba import njit
import shutil
import os

'''Integration of two 2D interfering waves via explicit method: u_tt=D*(u_xx+u_yy)'''

logging.basicConfig(level=logging.INFO)

time_steps=2000
N=200
L=2*np.pi*5
min=-L/2
max=L/2
x=np.linspace(min, max, N)
y=x
dx=x[1]-x[0]
dt=0.001
dy=dx
betax=dt/dx
betay=dt/dy
D=0.1
alphax=0.1
alphay=alphax

INTV=5
cc=0

w1=np.zeros((N,N))
w2=np.zeros((N, N))
w1_old=np.zeros((N,N))
w2_old=np.zeros((N, N))
w1_new=np.zeros((N,N))
w2_new=np.zeros((N,N))
w_totale=np.zeros((round(time_steps/INTV)+INTV, N, N))

for i in range(N):
    scaling=2 #regola lo scaling dell'onda
    xi=dx*i*scaling
    for j in range(N):
        yj=dy*j*scaling
        w1_old[i, j]= 2+np.sin(np.sqrt((xi-10)**2+(yj-10)**2))#np.exp(-((xi-12)**2+(yj-12)**2))
        w2_old[i, j]=np.sin(np.sqrt((xi-2)**2+(yj-2)**2))#np.exp(-((xi-5)**2+(yj-5)**2))

#initial condition on derivatives
w1[1:-1,1:-1]=w1_old[1:-1, 1:-1]+alphax*(w1_old[0:-2, 1:-1]-2*w1_old[1:-1,1:-1]+w1_old[2:,1:-1])+\
                alphay*(w1_old[1:-1,0:-2]-2*w1_old[1:-1,1:-1]+w1_old[1:-1,2:])
w2[1:-1,1:-1]=w2_old[1:-1, 1:-1]+alphax*(w2_old[0:-2, 1:-1]-2*w2_old[1:-1,1:-1]+w2_old[2:,1:-1])+\
                alphay*(w2_old[1:-1,0:-2]-2*w2_old[1:-1,1:-1]+w2_old[1:-1,2:])

@njit(fastmath=True, cache=True, parallel=True)
def calcolo_2D_waves(w1_old, w2_old, w1, w2, w1_new, w2_new):
    w1_new[1:-1,1:-1]=2*w1[1:-1,1:-1]-w1_old[1:-1,1:-1]+alphax*(w1[0:-2,1:-1]-2*w1[1:-1,1:-1]+w1[2:,1:-1])+\
            alphay*(w1[1:-1, 0:-2]-2*w1[1:-1,1:-1]+w1[1:-1,2:])

    w2_new[1:-1,1:-1]=2*w2[1:-1,1:-1]-w2_old[1:-1,1:-1]+alphax*(w2[0:-2,1:-1]-2*w2[1:-1,1:-1]+w2[2:,1:-1])+\
            alphay*(w2[1:-1, 0:-2]-2*w2[1:-1,1:-1]+w2[1:-1,2:])
    return w1_old, w2_old,  w1, w2, w1_new, w2_new

def boundary_conditions(kind, vec):
    if kind=='reflecting':
        vec[0, :]=vec[1, :]
        vec[N-1,:]=vec[N-2,:]
        vec[:,0]=vec[:,1]
        vec[:,N-1]=vec[:,N-2]
    if kind=='periodic':
        vec[0,:]=vec[N-2,:]
        vec[N-1,:]=vec[1,:]
        vec[:,0]=vec[:,N-2]
        vec[:, N-1]=vec[:,1]
    if kind=='absorbing':
        vec[0,:]=0
        vec[N-1,:]=0
        vec[:,0]=0
        vec[:, N-1]=0
    return vec

cammino='dati_2D'
h=dx
if os.path.exists(cammino)==True:
    shutil.rmtree(cammino)
os.mkdir(cammino)

def writeVtk(count):
    
    fp = open(f"{cammino}/data_" + str(count) + ".vtk", "w")
    fp.write("# vtk DataFile Version 4.1 \n")
    fp.write("COMMENT\n")
    fp.write("ASCII\n")
    fp.write("DATASET STRUCTURED_POINTS \n")
    fp.write("DIMENSIONS " + str(N) + " " + str(N) + " 1 \n")
    fp.write("ORIGIN 0 0 0\n")
    fp.write("SPACING " + str(h) + " " + str(h) + " 0 \n")
    fp.write("POINT_DATA " + str(N*N) + "\n")
    fp.write("SCALARS U double 1\n")
    fp.write("LOOKUP_TABLE default\n")
    for i in range(N):
        for j in range(N):
            fp.write(str(w_sum[i, j]) + "\n")
    fp.close()

execute=True
if execute==True:
    for t in range(time_steps):
        logging.info(f'step {t}')
        w1_old, w2_old, w1, w2, w1_new, w2_new=calcolo_2D_waves(w1_old, w2_old, w1, w2, w1_new, w2_new)

        w1_old[1:-1,1:-1]=w1[1:-1,1:-1]
        w1[1:-1,1:-1]=w1_new[1:-1,1:-1]

        w2_old[1:-1,1:-1]=w2[1:-1,1:-1]
        w2[1:-1,1:-1]=w2_new[1:-1,1:-1]

        kind='periodic'
        w1_old=boundary_conditions(kind, w1_old)
        w1=boundary_conditions(kind, w1)
        w1_new=boundary_conditions(kind, w1_new)

        w2_old=boundary_conditions(kind, w2_old)
        w2=boundary_conditions(kind, w2)
        w2_new=boundary_conditions(kind, w2_new)

        if t%INTV==0:
            w_totale[cc,:,:]=(w1_old+w2_old)
            w_sum=w1_old+w2_old
            writeVtk(cc)
            cc+=1
    graph.animate_matplotlib(x, y, w_totale)       



    



        