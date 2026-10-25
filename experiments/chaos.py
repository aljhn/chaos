import diffrax as dfx
import jax
import jax.numpy as jnp

from chaos_pinns.odes import MassSpringDamper

m = 1.0
k = 500.0
d = 5.0


def combined_dynamics(t, y, args):
    x = y[:2]
    U = y[2:].reshape(2, 2)

    J = jacobian(x)

    dx = state_dynamics(x)
    dU = J @ U

    return jnp.concatenate((dx, dU.flatten()))


T0 = 0.0
T1 = 50.0
dt_data = 0.01
iterations = int((T1 - T0) / dt_data)

x_initial = jnp.array([1.0, 0.0])
Q_initial = jnp.eye(2)

term = dfx.ODETerm(combined_dynamics)
solver = dfx.Tsit5()
controller = dfx.PIDController(rtol=1e-8, atol=1e-10)


def qr_positive_diagonal(Z):
    Q, R = jnp.linalg.qr(Z)
    signs = jnp.where(jnp.diag(R) < 0.0, -1.0, 1.0)
    Q = Q * signs[None, :]
    R = signs[:, None] * R
    return Q, R


@jax.jit
def compute_lyapunov_exponents(x0, Q0):
    def step(carry, t0):
        x, Q, log_sums = carry

        U_start = jnp.eye(2)

        y0 = jnp.concatenate((x, U_start.flatten()))

        solution = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t0 + dt_data,
            dt0=1e-3,
            y0=y0,
            stepsize_controller=controller,
        )

        y = solution.ys[-1]
        x_next = y[:2]

        M = solution.ys[-1, 2:].reshape(2, 2)
        Z = M @ Q

        Q_next, R = qr_positive_diagonal(Z)

        log_sums = log_sums + jnp.log(jnp.diag(R))

        return (x_next, Q_next, log_sums), None

    times = T0 + jnp.arange(iterations) * dt_data
    initial_carry = (x0, Q0, jnp.zeros(2))

    (_, _, log_sums), _ = jax.lax.scan(step, initial_carry, times)
    return log_sums / (iterations * dt_data)


lyapunov_exponents = compute_lyapunov_exponents(x_initial, Q_initial)
print(lyapunov_exponents)
