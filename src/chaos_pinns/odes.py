from abc import ABC, abstractmethod
from collections.abc import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class InitialValueProblem(ABC):
    x0: Float[Array, " dim"]
    t0: float
    t1: float

    solution: dfx.Solution | None

    def __init__(self, x0: Float[Array, " dim"], t0: float, t1: float):
        self.x0 = x0
        self.t0 = t0
        self.t1 = t1

        self.solution = None

    @abstractmethod
    def vector_field(self, t, x):
        pass

    def tangent_vector_field(self, t, x, U):
        vector_field_jacobian = jax.jacfwd(self.vector_field, argnums=1)
        J = vector_field_jacobian(t, x)
        return J @ U

    def solve(self, dt0: float = 1e-2):
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, x, args: self.vector_field(t, x)),
            dfx.Tsit5(),
            t0=self.t0,
            t1=self.t1,
            dt0=dt0,
            y0=self.x0,
            saveat=dfx.SaveAt(dense=True),
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1_000_000,
        )
        return solution

    def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch dim"]:
        if self.solution is None:
            self.solution = self.solve()

        return jax.vmap(self.solution.evaluate)(t[:, 0])

    def sample_validation(self, key: PRNGKeyArray, n: int) -> tuple[Float[Array, "batch 1"], Float[Array, "batch dim"]]:
        t = jax.random.uniform(key, shape=(n, 1), minval=self.t0, maxval=self.t1)
        x = self.get_true(t)
        return t, x

    def residual(
        self,
        model: Callable[[Float[Array, "1"]], Float[Array, " dim"]],
        t: Float[Array, "1"],
    ) -> Float[Array, " dim"]:
        x = model(t)
        dx = jax.jacfwd(model)(t)[:, 0]

        f = self.vector_field(t[0], x)

        residual = dx - f
        return residual


class MassSpringDamper(InitialValueProblem):
    A: Float[Array, "2 2"]

    def __init__(
        self,
        x0: Float[Array, "2"],
        t0: float,
        t1: float,
        m: float,
        d: float,
        k: float,
    ):
        self.A = jnp.array([[0.0, 1.0], [-k / m, -d / m]])
        super().__init__(x0, t0, t1)

    def vector_field(self, t, x):
        return self.A @ x


class Lorenz(InitialValueProblem):
    sigma: float
    rho: float
    beta: float

    def __init__(
        self,
        x0: Float[Array, "3"],
        t0: float,
        t1: float,
        sigma: float,
        rho: float,
        beta: float,
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        super().__init__(x0, t0, t1)

    def vector_field(self, t, x):
        x1, x2, x3 = x
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return jnp.array([dx1, dx2, dx3])


# class MassSpringDamper(InitialValueProblem):
#     m: float
#     d: float
#     k: float
#
#     x0: float
#     dx0: float
#
#     def __init__(
#         self,
#         m: float,
#         d: float,
#         k: float,
#         x0: float,
#         dx0: float,
#         t0: float,
#         t1: float,
#     ):
#         super().__init__(t0, t1)
#         self.m = m
#         self.d = d
#         self.k = k
#
#         self.x0 = x0
#         self.dx0 = dx0
#
#     def get_true(self, t: Float[Array, "batch 1"]) -> Float[Array, "batch 1"]:
#         alpha = self.d / (2.0 * self.m)
#         omega_squared = self.k / self.m
#         delta = alpha * alpha - omega_squared
#
#         tau = t - self.t0
#
#         C = 1.0
#         S = tau
#
#         eps = 1e-8
#         if delta > -eps and delta < eps:
#             delta_tau_tau = delta * tau * tau
#             C = 1.0 + delta_tau_tau / 2.0 + delta_tau_tau * delta_tau_tau / 24.0
#             S = tau * (1.0 + delta_tau_tau / 6.0 + delta_tau_tau * delta_tau_tau / 120.0)
#         elif delta < 0.0:
#             sqrt_minus_delta = jnp.sqrt(-delta)
#             C = jnp.cos(sqrt_minus_delta * tau)
#             S = jnp.sin(sqrt_minus_delta * tau) / sqrt_minus_delta
#         elif delta > 0.0:
#             sqrt_delta = jnp.sqrt(delta)
#             C = jnp.cosh(sqrt_delta * tau)
#             S = jnp.sinh(sqrt_delta * tau) / sqrt_delta
#
#         return jnp.exp(-alpha * tau) * (self.x0 * C + (self.dx0 + alpha * self.x0) * S)
#
#     def residual(
#         self,
#         model: Callable[[Float[Array, "1"]], Float[Array, "1"]],
#         t: Float[Array, "1"],
#     ) -> Float[Array, "1"]:
#         def state(t: Float[Array, ""]) -> Float[Array, ""]:
#             return model(t[None])[0]
#
#         tt = t[0]
#         x = state(tt)
#         dx = jax.grad(state)(tt)
#         ddx = jax.grad(jax.grad(state))(tt)
#
#         return self.m * ddx + self.d * dx + self.k * x
