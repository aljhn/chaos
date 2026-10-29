from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from chaos_pinns.collocation import CollocationSampler
from chaos_pinns.dataloading import TrainLoader, ValidationLoader

Model = Callable[[Float[Array, "batch input_size"]], Float[Array, "batch output_size"]]


def loss_data(
    model: Model,
    x: Float[Array, "batch input_size"],
    y: Float[Array, "batch output_size"],
) -> Float[Array, ""]:
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y_pred - y) ** 2.0)


def loss_physics(
    model: Model,
    x_c: Float[Array, "size dim"],
    physics_residual: Callable[[Model, Float[Array, "size dim"]], Float[Array, " size"]],
) -> Float[Array, ""]:
    residual = physics_residual(model, x_c)
    return jnp.mean(residual**2.0)


def loss_pinn(
    model: Model,
    x: Float[Array, "batch input_size"],
    y: Float[Array, "batch output_size"],
    x_c: Float[Array, "size dim"],
    physics_residual: Callable[[Model, Float[Array, "size dim"]], Float[Array, " size"]],
    physics_weight: float,
) -> Float[Array, ""]:
    loss = loss_data(model, x, y)
    loss += physics_weight * loss_physics(model, x_c, physics_residual)
    return loss


def evaluate(
    model: Model,
    validationloader: ValidationLoader,
) -> Float[Array, ""]:
    loss_total = jnp.array(0.0)
    samples = 0

    for x_batch, y_batch in validationloader.iter():
        batch_loss = loss_data(model, x_batch, y_batch)
        batch_size = x_batch.shape[0]

        loss_total += batch_loss * batch_size
        samples += batch_size

    return loss_total / samples


def train(
    model: Model,
    optimizer: optax.GradientTransformation,
    epochs: int,
    print_every: int,
    trainloader: TrainLoader,
    validationloader: ValidationLoader,
    key: PRNGKeyArray,
) -> Model:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: Model,
        opt_state: PyTree,
        x: Float[Array, "batch input_size"],
        y: Float[Array, "batch output_size"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_data)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(1, epochs + 1):
        loss_train = jnp.array(0.0)
        samples_train = 0

        key, subkey = jax.random.split(key)
        for x_batch, y_batch in trainloader.iter(subkey):
            model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch)

            batch_size = x_batch.shape[0]
            loss_train += batch_loss * batch_size
            samples_train += batch_size

        loss_train /= samples_train

        if epoch % print_every == 0:
            loss_validation = evaluate(model, validationloader)
            print(f"epoch={epoch:6d}, loss_train={loss_train.item():.5f}, loss_validation={loss_validation.item():.5f}")

    return model


# def train_pinn(
#     model: Model,
#     optimizer: optax.GradientTransformation,
#     epochs: int,
#     print_every: int,
#     trainloader: TrainLoader,
#     validationloader: ValidationLoader,
#     key: PRNGKeyArray,
#     collocation_sampler: CollocationSampler,
#     physics_residual: Callable[[Model, Float[Array, "size dim"]], Float[Array, " size"]],
#     physics_weight: float,
# ) -> tuple[Model, list, list]:
#     opt_state = optimizealsor.init(eqx.filter(model, eqx.is_array))
#
#     @eqx.filter_jit
#     def train_step(
#         model: Model,
#         opt_state: PyTree,
#         x: Float[Array, "batch input_size"],
#         y: Float[Array, "batch output_size"],
#         x_collocation: Float[Array, "size dim"],
#     ):
#         loss_value, grads = eqx.filter_value_and_grad(loss_pinn)(
#             model, x, y, x_collocation, physics_residual, physics_weight
#         )
#         updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
#         model = eqx.apply_updates(model, updates)
#         return model, opt_state, loss_value
#
#     train_losses = []
#     validation_losses = []
#
#     for epoch in range(1, epochs + 1):
#         loss_train = jnp.array(0.0)
#         samples_train = 0
#
#         key, subkey = jax.random.split(key)
#         for x_batch, y_batch in trainloader.iter(subkey):
#             key, subkey = jax.random.split(key)
#             x_c = collocation_sampler.sample(subkey)
#
#             model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch, x_c)
#
#             batch_size = x_batch.shape[0]
#             loss_train += batch_loss * batch_size
#             samples_train += batch_size
#
#         loss_train /= samples_train
#         train_losses.append(loss_train)
#
#         if print_every is not None and epoch % print_every == 0:
#             loss_validation = evaluate(model, validationloader)
#             validation_losses.append(loss_validation)
#
#             print(f"epoch={epoch:6d}, loss_train={loss_train.item():.5f}, loss_validation={loss_validation.item():.5f}")
#
#     return model, train_losses, validation_losses


def train_pinn(
    model: Model,
    optimizer: optax.GradientTransformation,
    epochs: int,
    print_every: int,
    trainloader: TrainLoader,
    validationloader: ValidationLoader,
    key: PRNGKeyArray,
    collocation_sampler: CollocationSampler,
    physics_residual,
    physics_weight: float,
    lbfgs_steps: int = 1000,
) -> tuple[Model, list, list]:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, x_collocation):
        loss_value, grads = eqx.filter_value_and_grad(loss_pinn)(
            model, x, y, x_collocation, physics_residual, physics_weight
        )
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    train_losses = []
    validation_losses = []

    # Adam phase: use the existing minibatch and resampling strategy.
    for epoch in range(1, epochs + 1):
        loss_total = jnp.array(0.0)
        samples_total = 0

        key, subkey = jax.random.split(key)
        for x_batch, y_batch in trainloader.iter(subkey):
            key, subkey = jax.random.split(key)
            x_c = collocation_sampler.sample(subkey)

            model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch, x_c)

            n = x_batch.shape[0]
            loss_total += batch_loss * n
            samples_total += n

        loss_epoch = loss_total / samples_total
        train_losses.append(loss_epoch)

        if print_every is not None and epoch % print_every == 0:
            val_loss = evaluate(model, validationloader)
            validation_losses.append(val_loss)
            print(f"Adam epoch={epoch:6d}, train={float(loss_epoch):.6g}, validation={float(val_loss):.6g}")

    if lbfgs_steps <= 0:
        return model, train_losses, validation_losses

    # Gather one fixed, full training dataset.
    key, subkey = jax.random.split(key)
    batches = list(trainloader.iter(subkey))
    x_full = jnp.concatenate([batch[0] for batch in batches], axis=0)
    y_full = jnp.concatenate([batch[1] for batch in batches], axis=0)

    # Fix collocation points so L-BFGS sees a deterministic objective.
    key, subkey = jax.random.split(key)
    x_c_fixed = collocation_sampler.sample(subkey)

    params = eqx.filter(model, eqx.is_array)
    static_model = eqx.filter(model, eqx.is_array, inverse=True)

    def full_loss(params):
        full_model = eqx.combine(params, static_model)
        return loss_pinn(
            full_model,
            x_full,
            y_full,
            x_c_fixed,
            physics_residual,
            physics_weight,
        )

    lbfgs = optax.lbfgs()
    lbfgs_state = lbfgs.init(params)

    value_and_grad = jax.value_and_grad(full_loss)

    @eqx.filter_jit
    def lbfgs_step(params, lbfgs_state):
        value, grads = value_and_grad(params)
        updates, lbfgs_state = lbfgs.update(
            grads,
            lbfgs_state,
            params,
            value=value,
            grad=grads,
            value_fn=full_loss,
        )
        params = optax.apply_updates(params, updates)
        return params, lbfgs_state, value

    for step in range(1, lbfgs_steps + 1):
        params, lbfgs_state, loss_value = lbfgs_step(params, lbfgs_state)
        model = eqx.combine(params, static_model)

        if step == 1 or step % max(1, print_every) == 0:
            val_loss = evaluate(model, validationloader)
            train_losses.append(loss_value)
            validation_losses.append(val_loss)
            print(f"L-BFGS step={step:6d}, train={float(loss_value):.6g}, validation={float(val_loss):.6g}")

    return model, train_losses, validation_losses
