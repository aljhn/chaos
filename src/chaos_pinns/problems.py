from abc import ABC, abstractmethod
from collections.abc import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class InitialBoundaryValueProblem(ABC):
    domain_boundary: Float[Array, "dims 2"]

    def __init__(self, domain_boundary: Float[Array, "dims 2"]):
        self.domain_boundary = domain_boundary

    @abstractmethod
    def get_true(self, position: Float[Array, "batch input_dim"]) -> Float[Array, "batch output_dim"]:
        pass

    @abstractmethod
    def sample_initial_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        pass

    @abstractmethod
    def sample_boundary_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        pass

    def sample_ib(
        self, key: PRNGKeyArray, n_i: int, n_b: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        key1, key2 = jax.random.split(key, 2)
        x_i, u_i = self.sample_initial_conditions(key1, n_i)
        x_b, u_b = self.sample_boundary_conditions(key2, n_b)
        x_ib = jnp.concatenate((x_i, x_b), axis=0)
        u_ib = jnp.concatenate((u_i, u_b), axis=0)
        return x_ib, u_ib

    def sample_validation(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        dims = self.domain_boundary.shape[0]
        domain_boundary_min = self.domain_boundary[:, 0]
        domain_boundary_max = self.domain_boundary[:, 1]
        x = jax.random.uniform(key, shape=(n, dims), minval=domain_boundary_min, maxval=domain_boundary_max)
        u = self.get_true(x)
        return x, u

    @abstractmethod
    def residual(
        self,
        model: Callable[[Float[Array, " input_dim"]], Float[Array, " output_dim"]],
        position: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        pass


class Heat1D(InitialBoundaryValueProblem):
    k: float
    t0: float
    t1: float
    x0: float
    x1: float

    def __init__(self, k: float, domain_boundary: Float[Array, "dims 2"]):
        super().__init__(domain_boundary)
        self.k = k
        self.t0 = self.domain_boundary[0, 0].item()
        self.t1 = self.domain_boundary[0, 1].item()
        self.x0 = self.domain_boundary[1, 0].item()
        self.x1 = self.domain_boundary[1, 1].item()

    def get_true(self, position: Float[Array, "batch input_dim"]) -> Float[Array, "batch output_dim"]:
        t = position[:, 0, None]
        x = position[:, 1, None]

        L = self.x1 - self.x0
        F = jnp.sin(jnp.pi * (x - self.x0) / L)
        G = jnp.exp(-((jnp.pi / L) ** 2.0) * self.k * (t - self.t0))
        return F * G

    def sample_initial_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        t = jnp.full((n,), self.t0)
        x = jax.random.uniform(key, shape=(n,), minval=self.x0, maxval=self.x1)
        tx = jnp.stack((t, x), axis=1)
        u = self.get_true(tx)
        return tx, u

    def sample_boundary_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        t = jax.random.uniform(key, shape=(n,), minval=self.t0, maxval=self.t1)
        n0 = n // 2
        n1 = n - n0
        x0 = jnp.full((n0,), self.x0)
        x1 = jnp.full((n1,), self.x1)
        x = jnp.concatenate((x0, x1), axis=0)
        tx = jnp.stack((t, x), axis=1)
        u = self.get_true(tx)
        return tx, u

    def residual(
        self,
        model: Callable[[Float[Array, " input_dim"]], Float[Array, " output_dim"]],
        position: Float[Array, " input_dim"],
    ) -> Float[Array, ""]:
        def state(position):
            return model(position)[0]

        u_grad = jax.grad(state)(position)
        u_hessian = jax.hessian(state)(position)

        u_t = u_grad[0]
        u_xx = u_hessian[1, 1]
        return u_t - self.k * u_xx


class InitialValueProblem(ABC):
    t0: float
    t1: float

    def __init__(self, t0: float, t1: float):
        self.t0 = t0
        self.t1 = t1

    @abstractmethod
    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch output_dim"]:
        pass

    def sample_validation(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch 1"], Float[Array, "batch output_dim"]]:
        t = jax.random.uniform(key, shape=(n, 1), minval=self.t0, maxval=self.t1)
        x = self.get_true(t)
        return t, x

    @abstractmethod
    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, " output_dim"]],
        t: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        pass


class MassSpringDamper(InitialValueProblem):
    m: float
    d: float
    k: float

    x0: float
    dx0: float

    def __init__(
        self,
        m: float,
        d: float,
        k: float,
        x0: float,
        dx0: float,
        t0: float,
        t1: float,
    ):
        super().__init__(t0, t1)
        self.m = m
        self.d = d
        self.k = k

        self.x0 = x0
        self.dx0 = dx0

    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch 1"]:
        alpha = self.d / (2.0 * self.m)
        omega_squared = self.k / self.m
        delta = alpha * alpha - omega_squared

        tau = t - self.t0

        C = 1.0
        S = tau

        eps = 1e-8
        if delta > -eps and delta < eps:
            delta_tau_tau = delta * tau * tau
            C = 1.0 + delta_tau_tau / 2.0 + delta_tau_tau * delta_tau_tau / 24.0
            S = tau * (1.0 + delta_tau_tau / 6.0 + delta_tau_tau * delta_tau_tau / 120.0)
        elif delta < 0.0:
            sqrt_minus_delta = jnp.sqrt(-delta)
            C = jnp.cos(sqrt_minus_delta * tau)
            S = jnp.sin(sqrt_minus_delta * tau) / sqrt_minus_delta
        elif delta > 0.0:
            sqrt_delta = jnp.sqrt(delta)
            C = jnp.cosh(sqrt_delta * tau)
            S = jnp.sinh(sqrt_delta * tau) / sqrt_delta

        return jnp.exp(-alpha * tau) * (self.x0 * C + (self.dx0 + alpha * self.x0) * S)

    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, "1"]],
        t: Float[Array, "1"],
    ) -> Float[Array, "1"]:
        def state(t: Float[Array, ""]) -> Float[Array, ""]:
            return model(t[None])[0]

        tt = t[0]
        x = state(tt)
        dx = jax.grad(state)(tt)
        ddx = jax.grad(jax.grad(state))(tt)

        return self.m * ddx + self.d * dx + self.k * x


class Lorenz(InitialValueProblem):
    sigma: float
    rho: float
    beta: float

    x0: Float[Array, "3"]

    def __init__(
        self,
        sigma: float,
        rho: float,
        beta: float,
        x0: Float[Array, "3"],
        t0: float,
        t1: float,
    ):
        super().__init__(t0, t1)

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x0 = x0

        dt0 = 0.01

        solution = dfx.diffeqsolve(
            dfx.ODETerm(self.vector_field),
            dfx.Tsit5(),
            t0=self.t0,
            t1=self.t1,
            dt0=dt0,
            y0=self.x0,
            saveat=dfx.SaveAt(dense=True),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1_000_000,
        )
        self.solution = solution

    def vector_field(self, t, x, args):
        x1, x2, x3 = x
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return jnp.array([dx1, dx2, dx3])

    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch 3"]:
        return jax.vmap(self.solution.evaluate)(t[:, 0])

    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, "3"]],
        t: Float[Array, "1"]
    ) -> Float[Array, "3"]:
        x = model(t)
        dx = jax.jacfwd(model)(t)[:, 0]

        f = self.vector_field(None, x, None)

        residual = dx - f 
        return residual
