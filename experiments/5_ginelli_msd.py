import diffrax as dfx
import jax
import jax.numpy as jnp

from chaos_pinns.odes import MassSpringDamper


def main():
    m = 1.0
    d = 5.0  # underdamped
    # d = (4.0 * 1.0 * 500.0)**0.5 # critically damped
    # d = 100.0 # overdamped
    k = 500.0

    T0 = 0.0
    T1 = 50.0
    dt = 1e-1
    iterations = int((T1 - T0) / dt) + 1
    t_all = jnp.linspace(T0, T1, iterations)
    t_intervals = (t_all[:-1], t_all[1:])

    dims = 2

    x0 = jnp.array([1.0, 0.0])
    Q0 = jnp.eye(dims)

    msd = MassSpringDamper(x0, T0, T1, m, d, k)

    def combined_dynamics(t, y, args):
        x = y[:dims]
        Q = y[dims:].reshape(dims, dims)
        dx = msd.vector_field(t, x)
        dQ = msd.tangent_vector_field(t, x, Q)
        return jnp.concatenate((dx, dQ.flatten()))

    term = dfx.ODETerm(combined_dynamics)
    solver = dfx.Tsit5()
    controller = dfx.PIDController(rtol=1e-8, atol=1e-10)

    def qr_positive_diagonal(Z):
        Q, R = jnp.linalg.qr(Z)
        signs = jnp.where(jnp.diag(R) < 0.0, -1.0, 1.0)
        Q = Q * signs[None, :]
        R = signs[:, None] * R
        return Q, R

    def step_forward(carry, t_interval):
        x, Q = carry
        y0 = jnp.concatenate((x, Q.flatten()), axis=0)
        t0, t1 = t_interval

        dt0 = 1e-2
        solution = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=controller,
        )

        x_next = solution.ys[-1, :dims]

        Z = solution.ys[-1, dims:].reshape(dims, dims)
        Q_next, R = qr_positive_diagonal(Z)

        carry_next = (x_next, Q_next)
        y = (x_next, Q_next, R)
        return carry_next, y

    initial_carry_forward = (x0, Q0)
    _, (x_all, Q_all, R_all) = jax.lax.scan(step_forward, initial_carry_forward, t_intervals)

    x_all = jnp.concatenate((x0[None, :], x_all), axis=0)
    Q_all = jnp.concatenate((Q0[None, :, :], Q_all), axis=0)

    skip_time = 0.1 * (T1 - T0)
    skip_iterations = int(skip_time / dt)
    log_sum = jnp.sum(jnp.log(jnp.diagonal(R_all[skip_iterations:, :, :], axis1=1, axis2=2)), axis=0)
    lyapunov_exponents = log_sum / (t_all[-1] - t_all[skip_iterations])
    print("Lyapunov exponents:", lyapunov_exponents)

    A = jnp.array([[0.0, 1.0], [-k / m, -d / m]])
    eigenvalues, _ = jnp.linalg.eig(A)
    print("Real part of eigenvalues:", jnp.real(eigenvalues))

    def step_backward(carry, R):
        C_next = carry
        C = jax.scipy.linalg.solve_triangular(R, C_next, lower=False)
        C = C / jnp.linalg.norm(C, axis=0, keepdims=True)
        return C, C

    C_next = jnp.eye(dims)

    initial_carry_backward = C_next
    _, C_all_reversed = jax.lax.scan(step_backward, initial_carry_backward, R_all[::-1, :, :])

    C_all = jnp.concatenate([C_all_reversed[::-1, :, :], C_next[None, :, :]], axis=0)

    V_all = Q_all @ C_all


if __name__ == "__main__":
    main()
