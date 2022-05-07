import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})
plot_kind='dynamic'



def animate_plot(L, x, u):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    ax = plt.axes(xlim=(-L/2, L/2), ylim=(0, 1.5))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,


    def animate(i):
        # x = np.linspace(0, 4, 1000)
        
        # y = np.sin(2 * np.pi * (x - 0.01 * i))
        y = u[i]
        line.set_data(x, y)
        return line,

    ani=FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    plt.show()

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