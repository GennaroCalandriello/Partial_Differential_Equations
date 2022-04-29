from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

'''This program resolves the ODE of the Van der Pol oscillator, where mus are the vector containing various value of the parameter that regulates the nonlinearity of the equation'''

def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x]

a, b = 0, 10

mus = [2.]
styles = ["-", "--", ":"]
t = linspace(a, b, 500)

for mu, style in zip(mus, styles):
    sol = solve_ivp(vdp, [a, b], [1, 0], t_eval=t)
    plt.plot(sol.y[0], -sol.y[1], style, c='blue')
# plt.show()
  
# make a little extra horizontal room for legend
plt.xlim([-3,3])    
plt.legend([f"a={m}" for m in mus])
plt.xlabel('y')
plt.ylabel('x')
plt.title('Ciclo limite ODE')
# plt.axes().set_aspect(1)
# plt.plot(sol.y[0], sol.y[1], style)
plt.show()