import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from control import lqr,ctrb

G = 1.
M = 10
m = 0.1

x0 = jnp.array([0.8, 0.,jnp.pi/2,1.])

def orbit(y):
    return jnp.array([y[1],y[0]*y[3]**2 - (G * M * m)/y[0]**2,y[3],(-2 * y[1] * y[3])/y[0]])

jacobian_fn = jax.jacfwd(orbit, argnums=0)  # it returns the function in charge of computing jacobian

A = jacobian_fn(x0)
B = jnp.array([0.,1.,0.,1.]).reshape(4,1)

w = jnp.array([0.8, 0.,jnp.pi/2,1.])

Q = jnp.eye(4)

R = jnp.array([1])

K, S, E = lqr(A, B, Q, R)

print(f"Controllability Matrix Rank: {jnp.linalg.matrix_rank(ctrb(A,B))}")

print(f"Control Matrix: {K}")
print(f"Eigenvalues of the closed loop: {E}")

u = lambda x: -K @ (x-w)

print(u(jnp.array([1.,0.,0.,1.])))


