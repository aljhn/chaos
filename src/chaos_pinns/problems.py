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
    domain_boundary: Float[Array, "2"]

    def __init__(self, domain_boundary: Float[Array, "2"]):
        self.domain_boundary = domain_boundary

    @abstractmethod
    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch output_dim"]:
        pass

    def sample_validation(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch 1"], Float[Array, "batch output_dim"]]:
        domain_boundary_min = self.domain_boundary[0, 0]
        domain_boundary_max = self.domain_boundary[0, 1]
        t = jax.random.uniform(key, shape=(n, 1), minval=domain_boundary_min, maxval=domain_boundary_max)
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
    t0: float
    t1: float

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
        domain_boundary = jnp.array([[t0, t1]])
        super().__init__(domain_boundary)
        self.m = m
        self.d = d
        self.k = k

        self.x0 = x0
        self.dx0 = dx0
        self.t0 = t0
        self.t1 = t1

    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch 1"]:
        alpha = self.d / (2.0 * self.m)
        omega_squared = self.k / self.m
        Delta = alpha * alpha - omega_squared

        tau = t - self.t0

        C = 1.0
        S = tau

        if Delta < 0:
            sqrt_minus_Delta = jnp.sqrt(-Delta)
            C = jnp.cos(sqrt_minus_Delta * tau)
            S = jnp.sin(sqrt_minus_Delta * tau) / sqrt_minus_Delta
        elif Delta > 0:
            sqrt_Delta = jnp.sqrt(Delta)
            C = jnp.cosh(sqrt_Delta * tau)
            S = jnp.sinh(sqrt_Delta * tau) / sqrt_Delta

        return jnp.exp(-alpha * tau) * (self.x0 * C + (self.dx0 + alpha * self.x0) * S)

    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, "1"]],
        t: Float[Array, "1"],
    ) -> Float[Array, "1"]:
        def state(t):
            return model(t)[0]

        x = state(t)[0]
        dx = jax.grad(state)(t)[0]
        ddx = jax.hessian(state)(t)[0, 0]

        return self.m * ddx + self.d * dx + self.k * x


class Lorenz(InitialValueProblem):
    sigma: float
    rho: float
    beta: float

    x0: Float[Array, "3"]
    t0: float
    t1: float

    @staticmethod
    def vector_field(t, x, args):
        x1, x2, x3 = x
        sigma, rho, beta = args
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return jnp.array([dx1, dx2, dx3])

    def __init__(
        self,
        sigma: float,
        rho: float,
        beta: float,
        x0: Float[Array, "3"],
        t0: float,
        t1: float,
    ):
        domain_boundary = jnp.array([[t0, t1]])
        super().__init__(domain_boundary)

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1

        dt0 = 0.01
        args = (self.sigma, self.rho, self.beta)

        solution = dfx.diffeqsolve(
            dfx.ODETerm(Lorenz.vector_field),
            dfx.Tsit5(),
            t0=self.t0,
            t1=self.t1,
            dt0=dt0,
            y0=self.x0,
            saveat=dfx.SaveAt(dense=True),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1_000_000,
            args=args
        )
        self.solution = solution

    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch 3"]:
        return jax.vmap(self.solution.evaluate)(t[:, 0])

    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, "3"]],
        t: Float[Array, "1"]
    ) -> Float[Array, "3"]:
        x = model(t)
        dx = jax.jacfwd(model)(t)[:, 0]

        args = (self.sigma, self.rho, self.beta)
        f = Lorenz.vector_field(None, x, args)

        residual = dx - f 
        return residual
