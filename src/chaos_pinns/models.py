import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    layers: list

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_hidden_layers + 1)

        self.layers = []
        self.layers.append(eqx.nn.Linear(input_size, hidden_size, key=keys[0]))
        self.layers.append(jnp.tanh)

        for i in range(1, num_hidden_layers):
            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]))
            self.layers.append(jnp.tanh)

        self.layers.append(eqx.nn.Linear(hidden_size, output_size, key=keys[-1]))

    def __call__(self, x: Float[Array, " input_size"]) -> Float[Array, " output_size"]:
        for layer in self.layers:
            x = layer(x)
        return x


class MLPWithNormalizedTime:
    network: MLP
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)

    def __init__(
        self,
        network: MLP,
        t0: float,
        t1: float,
    ):
        self.network = network
        self.t0 = t0
        self.t1 = t1

    def normalize_time(self, t: Float[Array, "1"]) -> Float[Array, "1"]:
        return 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0

    def __call__(self, t: Float[Array, "1"]) -> Float[Array, " dims"]:
        tau = self.normalize_time(t)
        z = self.network(tau)
        return z
