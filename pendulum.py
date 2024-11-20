import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from control import lqr,ctrb

# Pendulum parameters
g = 9.81  # Gravity
L = 1.0   # Length of the pendulum
theta0 = np.pi / 4  # Initial angle
omega0 = 0.0  # Initial angular velocity

# Time parameters
dt = 0.01  # Time step
t_end = 50.0  # End time
t = np.arange(0, t_end, dt)

#Control Matrices
B = np.array([0   ,1]).reshape(2,1)

w =np.array([np.pi,0])

A = np.array([[0,1],
            [g/L,0]])

Q = 10 * np.array([[1,0],
              [0,1]])

R = np.array([1])

K, S, E = lqr(A, B, Q, R)

print(f"Control Matrix: {K}")
print(f"Eigenvalues of the closed loop: {E}")

rng = np.random.default_rng(0)
# Solve the differential equation for the pendulum
def pendulum_ode(t, y, u,rng):
    theta, omega = y
    dy = np.array([omega, -g / L * np.sin(theta)]) + B @ u(y)
    dy[0] += rng.normal(0.,1.)
    return dy

u = lambda x: -K @ (x-w)

sol = solve_ivp(pendulum_ode, [0, t_end], [theta0, omega0], t_eval=t,args = (u,rng))

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Set equal scaling
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    x = L * np.sin(sol.y[0, i])
    y = -L * np.cos(sol.y[0, i])
    line.set_data([0, x], [0, y])
    return line,

ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)

plt.show()