import numpy as np
import matplotlib.pyplot as plt

'''This program executes the integration of u_t=a*u_x PDE through Lax and Lax Wendroff method. The integrator is regulated by the value of 'scheme'
'''
def animate_plot(x, u, scheme, alpha):
    plt.clf()
    plt.plot(x,u)
    if scheme==0:
        plt.gca().legend(('Lax $\\alpha$='+str(round(alpha,3)),''))
    if scheme==1:
        plt.gca().legend(('Lax Wendroff $\\alpha$='+str(round(alpha,3)),''))
    if scheme ==2:
        plt.gca().legend(('L-W con $\\partial^2$, $\\alpha$='+str(round(alpha,3)),''))
    plt.axis([xmin, xmax, -1, 1]) # IMPORTANTE
    plt.title('t='+str(round(dt*h,3)),fontsize=16)
    plt.xlabel('x',fontsize=18)
    plt.ylabel('u',fontsize=18)
    plt.draw()
    plt.pause(0.01)
    # plt.show()

if __name__=='__main__':

    scheme=1

    nx=100
    L=2*np.pi
    xmax=L
    xmin=-L
    dx=(xmax-xmin)/nx
    dt=0.1
    v=1.
    alpha=v*dt/dx
    x=np.arange(xmin, xmax, dx)
    u=np.exp(-x**2/2)*np.cos(2*np.pi*x)
    t_end=50
    nt=int(t_end/dt)
    t=np.linspace(0,t_end, nt)
    n=len(u) #lunghezza array

    if scheme ==0: #LAX
        unew=u
        for h in range (1, len(t)):
            for i in range(1, n-1):
                unew[i]=0.5*(u[i+1]+u[i-1])+alpha*(u[i+1]-u[i-1])
            unew[n-1]=0.5*(u[0]+u[n-2])+alpha*(u[0]-u[n-2])
            unew[0]=0.5*(u[1]+u[n-1])+alpha*(u[1]-u[n-1])
            animate_plot(x, u, 0, alpha)

    if scheme==1: #LAX WENDROFF
        unew=u
        for h in range(1, len(t)):
            for i in range(0, n-1):
                unew[i]=0.5*(u[i+1]+u[i])+0.5*alpha*(u[i+1]-u[i])
            unew[n-1]=0.5*(u[0]+u[n-1])+0.5*alpha*(u[0]-u[n-1])
            unew[0]=0.5*(u[1]+u[0])+0.5*alpha*(u[1]-u[0]) 
            animate_plot(x, u, 1, alpha)

    if scheme == 2: # LAX WENDROFF CON DERIVATA SECONDA 2 STEPS IN 1, pag 100 Numerical Pde
        unew=u
        for h in range(1, len(t)):
            for k in range(0, n-1): 
                unew[k]=u[k]-alpha*0.5*(u[k-1]-u[k+1])+0.5*alpha**2*(u[k-1]-2*u[k]+u[k+1])
            unew[n-1]=u[n-1]-alpha*0.5*(u[n-2]-u[0])+0.5*alpha**2*(u[n-2]-2*u[n-1]+u[0])
            unew[0]=u[0]-alpha*0.5*(u[n-1]-u[1])+0.5*alpha**2*(u[n-1]-2*u[0]+u[1])
            animate_plot(x, u, 2, alpha)
        print(u)
            