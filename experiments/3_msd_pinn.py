from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import PRNGKeyArray

from chaos_pinns.collocation import UniformCollocationSampler
from chaos_pinns.dataloading import TrainLoader, ValidationLoader
from chaos_pinns.models import MLP
from chaos_pinns.problems import MassSpringDamper, InitialValueProblem
from chaos_pinns.training import Model, train_pinn


@dataclass
class Config:
    seed: int = 42069

    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 64
    num_hidden_layers: int = 6

    m = 1.0
    c = 1.0
    k = 1.0

    x0 = 1.0
    dx0 = 1.0
    t0 = 0.0
    t1 = 10.0

    n_validation: int = 100
    n_collocation: int = 1000
    batch_size: int = 32

    learning_rate: float = 1e-3
    epochs: int = 5000
    print_every: int = 100
    physics_weight: float = 1e-1


def train_model(config: Config, problem: InitialValueProblem, key: PRNGKeyArray) -> Model:
    key, subkey = jax.random.split(key, 2)
    model = MLP(config.input_size, config.output_size, config.hidden_size, config.num_hidden_layers, subkey)

    t_i = jnp.array([[config.t0]])
    x_i = jnp.array([[config.x0]])

    key, subkey = jax.random.split(key, 2)
    tx_val, u_val = problem.sample_validation(subkey, config.n_validation)

    trainloader = TrainLoader(t_i, x_i, config.batch_size)
    validationloader = ValidationLoader(tx_val, u_val, config.batch_size)
    collocation_sampler = UniformCollocationSampler(config.n_collocation, problem.domain_boundary)

    def physics_residual(model, positions):
        return jax.vmap(problem.diffeq, in_axes=(None, 0))(model, positions)

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


def evaluate_model(model: Model, problem: InitialValueProblem) -> None:
    t0 = problem.domain_boundary[0, 0]
    t1 = problem.domain_boundary[0, 1]

    n = 100
    t = jnp.linspace(t0, t1, n)[:, None]
    x = jax.vmap(model)(t)

    x_true = problem.get_true(t)

    print("MSE:", jnp.mean((x - x_true) ** 2))

    plt.figure()
    plt.plot(t, x_true)
    plt.plot(t, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.show()


def main():
    config = Config()
    problem = MassSpringDamper(config.m, config.c, config.k, config.x0, config.dx0, config.t0, config.t1)

    key = jax.random.PRNGKey(config.seed)
    model = train_model(config, problem, key)

    evaluate_model(model, problem)


if __name__ == "__main__":
    main()
