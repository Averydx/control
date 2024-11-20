import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

import jax.numpy as jnp
from jax import jacrev

from control import lqr,ctrb,obsv,lqe

# Pendulum parameters
g = -9.81  # Gravity
m = 1. #mass of pendulum bob
M= 5. #Mass of cart
L = 0.5   # Length of the pendulum
delta = 1.


# Time parameters
dt = 0.01  # Time step
t_end = 10.0  # End time
t = np.arange(0, t_end, dt)

#Initial conditions
x0 = 0. 
v0 = 0. 
theta0 = np.pi/4
omega0 = 1.

def system_ctrl(y):
    x,v,theta,omega = y
 
    v_dot = -m**2 * L**2 * g * jnp.cos(theta) * jnp.sin(theta) + m * L**2 *(m * L * omega**2 * jnp.sin(theta) - delta * v)

    omega_dot = (m+M) * m * g * L * jnp.sin(theta) - m * L * jnp.cos(theta) * (m * L * omega**2 * jnp.sin(theta) - delta * v)

    denom = m * L ** 2 *(M + m*(1-jnp.cos(theta)**2))

    return jnp.array([v,v_dot/denom,omega,omega_dot/denom])

rng = np.random.default_rng(0)

#Control Matrices
w = jnp.array([0.,0.,0.,0.])

jacobian_fn = jacrev(system_ctrl)
A = jacobian_fn(w)
C = jnp.array([1.,0.,0.,0.])

###Kalman Filter Matrices
V_d = np.eye(4)
V_n = np.eye(1)
K_f,P,E = lqe(A,np.eye(4),C,V_d,V_n)

print(f"Gain Matrix: {K_f}")

#Solve the differential equation for the pendulum
def system(t, y):
    x,v,theta,omega,x_hat,v_hat,theta_hat,omega_hat = y
 
    v_dot = -m**2 * L**2 * g * np.cos(theta) * np.sin(theta) + m * L**2 *(m * L * omega**2 * np.sin(theta) - delta * v)
    omega_dot = (m+M) * m * g * L * np.sin(theta) - m * L * np.cos(theta) * (m * L * omega**2 * np.sin(theta) - delta * v)
    denom = m * L ** 2 *(M + m*(1-np.cos(theta)**2))

    dy_hat = A @ np.array([x_hat,v_hat,theta_hat,omega_hat]) + K_f @ np.array([x - x_hat])

    dy = np.array([v,v_dot/denom,omega,omega_dot/denom])

    return np.concatenate((dy,dy_hat))

sol = solve_ivp(system, [0, t_end], [x0,v0,theta0,omega0,0.,0.,0.,0.], t_eval=t)

# Create the plots
fig, ax = plt.subplots(ncols = 2, nrows = 2)
titles = ['x','v','$\\theta$','$\\omega$']
tick = 0
for i in range(2):
    for j in range(2):
        ax[i,j].set_title(f'{titles[tick]}')
        ax[i,j].plot(t,sol.y[tick,:])
        ax[i,j].plot(t,sol.y[tick+4,:],'--')
        tick += 1

plt.show()

#Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Set equal scaling
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

ax.plot(np.arange(-4,4,0.01),np.zeros_like(np.arange(-4,4,0.01)),color='r')

rect = patches.Rectangle((x0, 0.), 0.2, 0.2, linewidth=2, edgecolor='r', facecolor='r')
ax.add_patch(rect)

def animate(i):
    x = sol.y[0,i] + L * np.sin(sol.y[2, i])
    y = -L * np.cos(sol.y[2, i])
    line.set_data([sol.y[0,i], x], [0, y])
    rect.set_xy((sol.y[0,i] - 0.1,-0.1))
    return line,rect

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()