import diffrax as dfx
import jax
import jax.numpy as jnp

from chaos_pinns.odes import MassSpringDamper

m = 1.0
d = 5.0
k = 500.0

T0 = 0.0
T1 = 50.0
dt = 1e-1
iterations = int((T1 - T0) / dt) + 1
times = jnp.linspace(T0, T1, iterations)

x0 = jnp.array([1.0, 0.0])
Q0 = jnp.eye(2)

msd = MassSpringDamper(x0, T0, T1, m, d, k)


def combined_dynamics(t, y, args):
    x = y[:2]
    U = y[2:].reshape(2, 2)
    dx = msd.vector_field(t, x)
    dU = msd.tangent_vector_field(t, x, U)
    return jnp.concatenate((dx, dU.flatten()))


term = dfx.ODETerm(combined_dynamics)
solver = dfx.Tsit5()
controller = dfx.PIDController(rtol=1e-8, atol=1e-10)


def qr_positive_diagonal(Z):
    Q, R = jnp.linalg.qr(Z)
    signs = jnp.where(jnp.diag(R) < 0.0, -1.0, 1.0)
    Q = Q * signs[None, :]
    R = signs[:, None] * R
    return Q, R


def step(carry, t):
    x, Q, log_sums = carry
    U_start = jnp.eye(2)
    y0 = jnp.concatenate((x, U_start.flatten()), axis=0)

    dt0 = 1e-3
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t,
        t1=t + dt,
        dt0=dt0,
        y0=y0,
        stepsize_controller=controller,
    )

    x_next = solution.ys[-1, :2]

    M = solution.ys[-1, 2:].reshape(2, 2)
    Z = M @ Q
    Q_next, R = qr_positive_diagonal(Z)

    log_sums = log_sums + jnp.log(jnp.diag(R))

    carry = (x_next, Q_next, log_sums)
    y = (x_next, Q_next)
    return carry, y

initial_carry = (x0, Q0, jnp.zeros(2))

(_, _, log_sum), (x_all, Q_all) = jax.lax.scan(step, initial_carry, times)
lyapunov_exponents = log_sum / (iterations * dt)
print(lyapunov_exponents)
