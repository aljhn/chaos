from dataclasses import dataclass
from pathlib import Path

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from chaos_pinns.collocation import UniformCollocationSampler
from chaos_pinns.dataloading import TrainLoader, ValidationLoader
from chaos_pinns.models import MLP, MLPWithNormalizedTime
from chaos_pinns.odes import InitialValueProblem, Lorenz
from chaos_pinns.training import Model, train_pinn


@dataclass
class Config:
    seed: int = 42069

    input_size: int = 1
    output_size: int = 3
    hidden_size: int = 64
    num_hidden_layers: int = 5

    dims: int = 3

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    x0_1: float = 1.0
    x0_2: float = 1.0
    x0_3: float = 1.0
    t0: float = 0.0
    t1: float = 4.0

    data_t0: float = 0.0
    data_t1: float = 100.0
    data_dt: float = 1e-3

    n_validation: int = 128
    n_collocation: int = 2048
    batch_size: int = 32

    learning_rate: float = 1e-3
    epochs: int = 20000
    print_every: int = 500
    physics_weight: float = 1e-4


def get_lyapunov_stuff(config: Config, system: InitialValueProblem):
    iterations = int((config.data_t1 - config.data_t0) / config.data_dt) + 1
    t_all = jnp.linspace(config.data_t0, config.data_t1, iterations)
    t_intervals = (t_all[:-1], t_all[1:])

    x0 = system.x0
    Q0 = jnp.eye(config.dims)

    def combined_dynamics(t, y, args):
        x = y[: config.dims]
        Q = y[config.dims :].reshape(config.dims, config.dims)
        dx = system.vector_field(t, x)
        dQ = system.tangent_vector_field(t, x, Q)
        return jnp.concatenate((dx, dQ.flatten()))

    term = dfx.ODETerm(combined_dynamics)
    solver = dfx.Tsit5()
    controller = dfx.PIDController(rtol=1e-7, atol=1e-9)

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
            max_steps=10_000_000,
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

    skip_time = 0.1 * (config.data_t1 - config.data_t0)
    skip_iterations = int(skip_time / config.data_dt)
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

    iterations = int((config.data_t1 - config.data_t0) / config.data_dt) + 1
    t_all = t_all[:iterations]
    x_all = x_all[:iterations]
    V_all = V_all[:iterations]

    return t_all, x_all, V_all, lyapunov_exponents


def train_model(
    config: Config,
    problem: InitialValueProblem,
    data: tuple[Float[Array, "batch 1"], Float[Array, "batch 3"]],
    key: PRNGKeyArray,
) -> tuple[Model, list, list]:
    key, subkey = jax.random.split(key, 2)
    model = MLP(config.input_size, config.output_size, config.hidden_size, config.num_hidden_layers, subkey)
    # model = MLPWithNormalizedTime(model, problem.t0, problem.t1)

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
    model, train_losses, validation_losses = train_pinn(
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

    return model, train_losses, validation_losses


def normalize_and_align_clvs(V: Float[Array, "time dims dims"]):
    """Normalize each CLV and remove arbitrary sign flips between time steps."""
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)
    V = V / jnp.maximum(norms, 1e-12)

    def align(previous, current):
        signs = jnp.where(
            jnp.sum(previous * current, axis=0) < 0.0,
            -1.0,
            1.0,
        )
        current = current * signs[None, :]
        return current, current

    _, V_tail = jax.lax.scan(align, V[0], V[1:])
    return jnp.concatenate((V[:1], V_tail), axis=0)


def train_model_clv(
    config: Config,
    problem: InitialValueProblem,
    data: tuple[Float[Array, "batch 1"], Float[Array, "batch 2"]],
    key: PRNGKeyArray,
    t_clv: Float[Array, " time"],
    V_clv: Float[Array, "time dims dims"],
    lyapunov_exponents: Float[Array, " dims"],
    clv_beta: float = 1.0,
) -> tuple[Model, list, list]:
    # Initialize model.
    key, subkey = jax.random.split(key)
    model = MLP(
        config.input_size,
        config.output_size,
        config.hidden_size,
        config.num_hidden_layers,
        subkey,
    )
    model = MLPWithNormalizedTime(model, problem.t0, problem.t1)

    # Validation data.
    key, subkey = jax.random.split(key)
    t_val, x_val = problem.sample_validation(subkey, config.n_validation)

    t_data, x_data = data
    trainloader = TrainLoader(t_data, x_data, config.batch_size)
    validationloader = ValidationLoader(t_val, x_val, config.batch_size)

    # Collocation sampler.
    domain_boundary = jnp.array([[problem.t0, problem.t1]])
    collocation_sampler = UniformCollocationSampler(
        config.n_collocation,
        domain_boundary,
    )

    # Prepare CLVs once, outside the training loop.
    t_clv = jnp.asarray(t_clv).reshape(-1)
    V_clv = normalize_and_align_clvs(V_clv)

    dims = V_clv.shape[-1]
    if V_clv.shape[1] != dims:
        raise ValueError("Each CLV matrix must be square.")
    if lyapunov_exponents.shape != (dims,):
        raise ValueError("Expected one Lyapunov exponent per CLV.")
    if dims != config.output_size:
        raise ValueError("The CLV dimension must match the PINN's state-space residual dimension.")

    # Diagnose whether the CLV-coordinate transformation is reliable.
    condition_numbers = jnp.linalg.cond(V_clv)
    print(
        "CLV condition numbers:",
        f"median={float(jnp.median(condition_numbers)):.3e},",
        f"p99={float(jnp.percentile(condition_numbers, 99)):.3e},",
        f"max={float(jnp.max(condition_numbers)):.3e}",
    )

    if float(jnp.max(condition_numbers)) > 1e8:
        print(
            "Warning: Some CLV matrices are severely ill-conditioned. "
            "The CLV-coordinate residual may be numerically unreliable."
        )

    # Mean-one weights: beta=0 gives uniform weights.
    weights = dims * jax.nn.softmax(clv_beta * lyapunov_exponents)
    sqrt_weights = jnp.sqrt(weights)

    # Flatten the matrices so each component can be interpolated in time.
    V_flat = V_clv.reshape(V_clv.shape[0], dims * dims)

    def interpolate_clv(t):
        interpolated = jax.vmap(
            lambda values: jnp.interp(t, t_clv, values),
            in_axes=1,
        )(V_flat)
        return interpolated.reshape(dims, dims)

    def physics_residual(model, positions):
        # Standard ODE residual: (batch, dims).
        residuals = jax.vmap(
            problem.residual,
            in_axes=(None, 0),
        )(model, positions)

        # Interpolate the CLV basis at each collocation time.
        times = positions[:, 0]
        V_batch = jax.vmap(interpolate_clv)(times)

        # Express residuals in CLV coordinates: V(t) a(t) = r(t).
        coefficients = jax.vmap(
            lambda V, r: jnp.linalg.solve(V, r),
        )(V_batch, residuals)

        # The usual mean-squared residual of this output becomes
        # mean_t [sum_i weights[i] * coefficients[t, i]**2 / dims].
        return coefficients * sqrt_weights[None, :]

    optimizer = optax.adamw(config.learning_rate)

    key, subkey = jax.random.split(key)
    model, train_losses, validation_losses = train_pinn(
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

    return model, train_losses, validation_losses


def plot_models(
    model1: Model,
    model2: Model,
    problem: InitialValueProblem,
    data_eval: tuple[Float[Array, "batch 1"], Float[Array, "batch 2"]],
    data_train: tuple[Float[Array, "batch 1"], Float[Array, "batch 2"]],
) -> None:
    t_eval, x_eval = data_eval
    t_train, x_train = data_train

    x1 = jax.vmap(model1)(t_eval)
    # x2 = jax.vmap(model2)(t_eval)

    print("MSE 1:", jnp.mean((x1 - x_eval) ** 2))
    # print("MSE 2:", jnp.mean((x2 - x_eval) ** 2))

    plt.figure()
    plt.plot(t_eval, x_eval[:, 0])
    plt.plot(t_eval, x1[:, 0])
    # plt.plot(t, x2[:, 0])
    plt.scatter(t_train, x_train[:, 0])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.show()


def plot_losses(loss_1: list, loss_2: list):
    n_epochs_1 = len(loss_1)
    epochs_1 = jnp.arange(1, n_epochs_1 + 1, 1)

    n_epochs_2 = len(loss_2)
    epochs_2 = jnp.arange(1, n_epochs_2 + 1, 1)

    plt.figure()
    plt.plot(epochs_1, loss_1)
    plt.plot(epochs_2, loss_2)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def main():
    config = Config()

    x0 = jnp.array([config.x0_1, config.x0_2, config.x0_3])
    system = Lorenz(x0, config.t0, config.t1, config.sigma, config.rho, config.beta)

    filename = "lorenz.npz"
    path = Path(filename)
    if path.exists():
        with np.load(path) as data:
            t_all = data["t_all"]
            x_all = data["x_all"]
            V_all = data["V_all"]
            lyapunov_exponents = data["lyapunov_exponents"]
    else:
        t_all, x_all, V_all, lyapunov_exponents = get_lyapunov_stuff(config, system)
        np.savez_compressed(
            filename,
            t_all=np.asarray(t_all),
            x_all=np.asarray(x_all),
            V_all=np.asarray(V_all),
            lyapunov_exponents=np.asarray(lyapunov_exponents),
        )

    n_all = t_all.shape[0]
    index_start = 0
    index_stop = int(n_all * (config.t1 / 2.0) / config.data_t1) + 1
    index_step = 100
    t_train = t_all[index_start:index_stop:index_step, None]
    x_train = x_all[index_start:index_stop:index_step, :]
    data_train = (t_train, x_train)

    key = jax.random.PRNGKey(config.seed)
    model, train_losses, validation_losses = train_model(config, system, data_train, key)

    # model_clv, train_losses_clv, validation_losses_clv = train_model_clv(
    #     config, system, data, key, t_all, V_all, lyapunov_exponents, clv_beta=1.0
    # )
    model_clv = None

    index_stop = int(n_all * config.t1 / config.data_t1) + 1
    index_step = 10
    t_eval = t_all[index_start:index_stop:index_step, None]
    x_eval = x_all[index_start:index_stop:index_step, :]
    data_eval = (t_eval, x_eval)

    plot_models(model, model_clv, system, data_eval, data_train)
    # plot_losses(train_losses, train_losses_clv)
    # plot_losses(validation_losses, validation_losses_clv)


if __name__ == "__main__":
    main()
