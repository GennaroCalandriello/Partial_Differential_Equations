import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
from matplotlib import animation

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

def animate_plot(L, x, u):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    ax = plt.axes(xlim=(-L/2, L/2), ylim=(-1.5,3))
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,


    def animate(i):
        y=(u[i])
        line.set_data(x, y)
        return line,

    ani=FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    fig.suptitle('Exp wave plot', fontsize=14)
    plt.show()
    # writervideo = animation.FFMpegWriter(fps=60)
    ani.save('Korteweg_de_Vries.gif')
    

def static_plot(u, x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j in range(u.shape[0]):
        ys = j*np.ones(u.shape[1])
        ax.plot(x, ys, u[j, :])

    # Image plot
    plt.figure()
    plt.imshow(np.flipud(u), aspect=8)
    plt.axis('off')
    plt.show()