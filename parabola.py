import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from control import lqr,ctrb
from jax import jacfwd
import jax.numpy as jnp


# Time parameters
dt = 0.01  # Time step
t_end = 1000.0  # End time
t = np.arange(0, t_end, dt)

#Model Parameters
g = 1.
a = 1.

####All below is the controller design
B = jnp.array([0.,1.]).reshape(2,1)


def parabola_ode_ctrl(y):
    x, v_x = y
    dy = jnp.array([v_x,(-4 * a**2 * x * v_x**2 - 2 * g * a * x)/(1+4*a**2 * x**2)])
    return dy

jacobian_fn = jacfwd(parabola_ode_ctrl, argnums=0)  # it returns the function in charge of computing jacobian

w = jnp.array([0.,0.])
A = jacobian_fn(w)

Q = jnp.eye(2)

R = jnp.array([1])

K, S, E = lqr(A, B, Q, R)

print(f"Controllability Matrix Rank: {jnp.linalg.matrix_rank(ctrb(A,B))}")

print(f"Control Matrix: {K}")
print(f"Eigenvalues of the closed loop: {E}")

# Solve the differential equation for the parabola
def parabola_ode(t, y, u):
    x, v_x = y
    dy = np.array([v_x,(-4 * a**2 * x * v_x**2 - 2 * g * a * x)/(1+4*a**2 * x**2)]) + B @ u(y)
    return dy

u = lambda x: -K @ (x-w)

sol = solve_ivp(parabola_ode, [0, t_end], [1, 0], t_eval=t,args = (u,))

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-2., 2.)
ax.set_ylim(-0.5, 2.0)


line, = ax.plot(np.arange(-2,2,0.01), np.arange(-2,2,0.01)**2)

particle = plt.Circle((0, 0), 0.1, color='r')
ax.add_patch(particle)

# Set equal scaling
ax.set_aspect('equal')

def animate(i):
    particle.center = (sol.y[0,i], sol.y[0,i]**2) 
    return particle,

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()