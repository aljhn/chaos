from dataclasses import dataclass

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from chaos_pinns.collocation import UniformCollocationSampler
from chaos_pinns.dataloading import TrainLoader, ValidationLoader
from chaos_pinns.models import MLP
from chaos_pinns.odes import InitialValueProblem, MassSpringDamper, MassSpringDamper1D
from chaos_pinns.training import Model, train_pinn


@dataclass
class Config:
    seed: int = 42069

    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 32
    num_hidden_layers: int = 4

    dims: int = 2

    m: float = 1.0
    d: float = 5.0
    k: float = 500.0

    x0: float = 1.0
    dx0: float = 0.0
    t0: float = 0.0
    t1: float = 1.0
    dt: float = 1e-3

    n_validation: int = 100
    n_collocation: int = 100
    batch_size: int = 32

    learning_rate: float = 1e-3
    epochs: int = 10000
    print_every: int = 1000
    physics_weight: float = 1e-4


def get_lyapunov_stuff(config: Config, system: InitialValueProblem):
    t1_original = config.t1
    config.t1 *= 10.0

    iterations = int((config.t1 - config.t0) / config.dt) + 1
    t_all = jnp.linspace(config.t0, config.t1, iterations)
    t_intervals = (t_all[:-1], t_all[1:])

    x0 = jnp.array([1.0, 0.0])
    Q0 = jnp.eye(config.dims)

    def combined_dynamics(t, y, args):
        x = y[: config.dims]
        Q = y[config.dims :].reshape(config.dims, config.dims)
        dx = system.vector_field(t, x)
        dQ = system.tangent_vector_field(t, x, Q)
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

        x_next = solution.ys[-1, : config.dims]

        Z = solution.ys[-1, config.dims :].reshape(config.dims, config.dims)
        Q_next, R = qr_positive_diagonal(Z)

        carry_next = (x_next, Q_next)
        y = (x_next, Q_next, R)
        return carry_next, y

    initial_carry_forward = (x0, Q0)
    _, (x_all, Q_all, R_all) = jax.lax.scan(step_forward, initial_carry_forward, t_intervals)

    x_all = jnp.concatenate((x0[None, :], x_all), axis=0)
    Q_all = jnp.concatenate((Q0[None, :, :], Q_all), axis=0)

    skip_time = 0.1 * (config.t1 - config.t0)
    skip_iterations = int(skip_time / config.dt)
    log_sum = jnp.sum(jnp.log(jnp.diagonal(R_all[skip_iterations:, :, :], axis1=1, axis2=2)), axis=0)
    lyapunov_exponents = log_sum / (t_all[-1] - t_all[skip_iterations])
    print("Lyapunov exponents:", lyapunov_exponents)

    def step_backward(carry, R):
        C_next = carry
        C = jax.scipy.linalg.solve_triangular(R, C_next, lower=False)
        C = C / jnp.linalg.norm(C, axis=0, keepdims=True)
        return C, C

    C_next = jnp.eye(config.dims)

    initial_carry_backward = C_next
    _, C_all_reversed = jax.lax.scan(step_backward, initial_carry_backward, R_all[::-1, :, :])

    C_all = jnp.concatenate([C_all_reversed[::-1, :, :], C_next[None, :, :]], axis=0)

    V_all = Q_all @ C_all

    config.t1 = t1_original
    iterations = int((config.t1 - config.t0) / config.dt) + 1
    t_all = t_all[:iterations]
    x_all = x_all[:iterations]
    V_all = V_all[:iterations]

    return t_all, x_all, V_all, lyapunov_exponents


def train_model(
    config: Config,
    problem: InitialValueProblem,
    data: tuple[Float[Array, "batch 1"], Float[Array, "batch 2"]],
    key: PRNGKeyArray,
) -> Model:
    key, subkey = jax.random.split(key, 2)
    model = MLP(config.input_size, config.output_size, config.hidden_size, config.num_hidden_layers, subkey)

    key, subkey = jax.random.split(key, 2)
    t_val, x_val = problem.sample_validation(subkey, config.n_validation)

    t_data, x_data = data
    trainloader = TrainLoader(t_data, x_data, config.batch_size)
    validationloader = ValidationLoader(t_val, x_val, config.batch_size)

    domain_boundary = jnp.array([[problem.t0, problem.t1]])
    collocation_sampler = UniformCollocationSampler(config.n_collocation, domain_boundary)

    def physics_residual(model, positions):
        return jax.vmap(problem.residual, in_axes=(None, 0))(model, positions)

    optimizer = optax.adamw(config.learning_rate)

    key, subkey = jax.random.split(key, 2)
    model = train_pinn(
        model,
        optimizer,
        config.epochs,
        config.print_every,
        trainloader,
        validationloader,
        subkey,
        collocation_sampler,
        physics_residual,
        config.physics_weight,
    )

    return model


def evaluate_model(
    model: Model,
    problem: InitialValueProblem,
    data: tuple[Float[Array, "batch 1"], Float[Array, "batch 2"]],
) -> None:
    t0 = problem.t0
    t1 = problem.t1

    n = 1000
    t = jnp.linspace(t0, t1, n)[:, None]
    x = jax.vmap(model)(t)

    x_true = problem.get_true(t)

    print("MSE:", jnp.mean((x - x_true) ** 2))

    t_data, x_data = data

    plt.figure()
    plt.plot(t, x_true[:, 0])
    plt.plot(t, x[:, 0])
    plt.scatter(t_data, x_data)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.show()


def main():
    config = Config()

    config.dims = 2
    x0 = jnp.array([config.x0, config.dx0])
    system = MassSpringDamper(x0, config.t0, config.t1, config.m, config.d, config.k)

    t_all, x_all, V_all, lyapunov_exponents = get_lyapunov_stuff(config, system)

    n_all = t_all.shape[0]
    data_percentage = 40
    data_points = 10

    index_start = 0
    index_stop = (n_all * data_percentage) // 100
    index_step = (index_stop - index_start) // data_points

    t_data = t_all[index_start:index_stop:index_step, None]
    x_data = x_all[index_start:index_stop:index_step, 0, None]
    data = (t_data, x_data)

    x0 = x_all[0, 0]
    dx0 = x_all[0, 1]

    config.dims = 1

    problem = MassSpringDamper1D(config.m, config.d, config.k, config.x0, config.dx0, config.t0, config.t1)

    key = jax.random.PRNGKey(config.seed)
    model = train_model(config, problem, data, key)

    evaluate_model(model, problem, data)


if __name__ == "__main__":
    main()
