import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


#model parameters
a = 0.01

r0 = 0.
dr0 = 1.
theta0 = 0. 
dtheta0 = 5.

# Time parameters
dt = 0.01  # Time step
t_end = 10.0  # End time
t = np.arange(0, t_end, dt)

def spiral_ode(t,y):
    r,v_r,theta,v_theta = y
    a_theta = (-theta/(1+theta**2)) * v_theta**2

    return np.array([v_r,a*a_theta,v_theta,a_theta])

sol = solve_ivp(spiral_ode, [0, t_end], [r0,dr0,theta0,dtheta0], t_eval=t)

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-4., 4.)
ax.set_ylim(-4., 4.)

# Set equal scaling
ax.set_aspect('equal')

line_data_x = []
line_data_y = []
line, = ax.plot([], [])

p = plt.Circle((r0 * np.sin(theta0), r0 * np.cos(theta0)), 0.05, color='b')
ax.add_patch(p)

def animate(i):
    x = sol.y[0, i] * np.sin(sol.y[2, i])
    y = sol.y[0, i] * np.cos(sol.y[2, i])
    p.center = (x, y) 
    line_data_x.append(x)
    line_data_y.append(y)
    line.set_data(line_data_x, line_data_y)
    return p,line

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()

