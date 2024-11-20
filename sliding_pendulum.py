import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from control import lqr,ctrb
from jax import jacfwd
import jax.numpy as jnp

# Pendulum parameters
g = 9.81  # Gravity
L = 1.0   # Length of the pendulum
m1=1.
m2 =1.

x0 = -0.5
dx0 = -0.5
theta0 = np.pi/2 # Initial angle
omega0 = 5.0  # Initial angular velocity

# Time parameters
dt = 0.01  # Time step
t_end = 10.0  # End time
t = np.arange(0, t_end, dt)

def pendulum_ode_ctrl(y):
    x,v_x,theta,v_theta = y

    a_x = (g * m2 * jnp.sin(theta) * jnp.cos(theta) + m2 * L * (v_theta ** 2) * jnp.sin(theta))/((m1 + m2) - m2 * (jnp.cos(theta))**2)
    a_theta = -a_x * jnp.cos(theta) - g * jnp.sin(theta)

    return jnp.array([v_x,a_x,v_theta,a_theta])


u = lambda x: -K @ (x-w)
jacobian_fn = jacfwd(pendulum_ode_ctrl, argnums=0)  # it returns the function in charge of computing jacobian

w = jnp.array([-1.0,0.,np.pi,0.])
A = jacobian_fn(w)

Q = jnp.eye(4)

R = 1 * jnp.array([1])

B = jnp.array([0.,1.,0.,1.]).reshape(4,1)

K, S, E = lqr(A, B, Q, R)

def pendulum_ode(t, y):
    x,v_x,theta,v_theta = y

    a_x = (g * m2 * np.sin(theta) * np.cos(theta) + m2 * L * (v_theta ** 2) * np.sin(theta))/((m1 + m2) - m2 * (np.cos(theta))**2)
    a_theta = -a_x * np.cos(theta) - g * np.sin(theta)

    return np.array([v_x,a_x,v_theta,a_theta]) + B @ u(y)

print(f"Controllability Matrix Rank: {jnp.linalg.matrix_rank(ctrb(A,B))}")

print(f"Control Matrix: {K}")
print(f"Eigenvalues of the closed loop: {E}")

sol = solve_ivp(pendulum_ode, [0, t_end], [x0,dx0,theta0,omega0], t_eval=t)

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-4., 4.)
ax.set_ylim(-4., 4.)

# Set equal scaling
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

ax.plot(np.arange(-4,4,0.01),np.zeros_like(np.arange(-4,4,0.01)),color='r')

def animate(i):
    x2 = sol.y[0, i] + L * np.sin(sol.y[2, i])
    y2 = -L * np.cos(sol.y[2, i])
    line.set_data([sol.y[0, i],x2], [0, y2])
    return line,

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()