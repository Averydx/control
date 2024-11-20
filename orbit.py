import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from control import lqr,ctrb
from jax import jacfwd

G = 1.
M = 10
m = 0.1

def orbit(y):
    return jnp.array([y[1],y[0]*y[3]**2 - (G * M * m)/y[0]**2,y[3],(-2 * y[1] * y[3])/y[0]])

# Time parameters
dt = 0.01  # Time step
t_end = 100.0  # End time
t = jnp.arange(0, t_end, dt)

x0 = jnp.array([0.7, 0.,jnp.pi,1.])

jacobian_fn = jacfwd(orbit, argnums=0)  # it returns the function in charge of computing jacobian

A = jacobian_fn(x0)
B = jnp.array([0.,1.,1.,0.]).reshape(4,1)

w = jnp.array([0.8, 0.,jnp.pi/2,0.5])

Q = jnp.eye(4)

R = jnp.array([1.])

K, S, E = lqr(A, B, Q, R)

def orbit_ctrl(y,u):
    return jnp.array([y[1],y[0]*y[3]**2 - (G * M * m)/y[0]**2,y[3],(-2 * y[1] * y[3])/y[0]]) + B @ u(y)

u = lambda x: -K @ (x-w)

sol = solve_ivp(lambda t,y,u: orbit_ctrl(y,u), [0, t_end], x0, t_eval=t,args = (u,))

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Set equal scaling
ax.set_aspect('equal')

# Create the circle patch initially at a fixed position
planet = plt.Circle((0, 0), 0.1, color='r')
ax.add_patch(planet)

sat = plt.Circle((x0[0], x0[2]), 0.05, color='b')
ax.add_patch(sat)

line_data_x = []
line_data_y = []
line, = ax.plot([], [])

def animate(i):
    x = sol.y[0, i] * jnp.cos(sol.y[2, i])
    y = sol.y[0, i] * jnp.sin(sol.y[2, i])
    print(sol.y[:, i])
    sat.center = (x, y) 
    line_data_x.append(x)
    line_data_y.append(y)
    line.set_data(line_data_x, line_data_y)

    return sat,line 

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()