"""Separate edit-policy objective gradients using a shared stochastic forward pass."""
import jax
import jax.numpy as jnp
import optax


def edit_component_gradients(loss_terms_fn, params):
    """Differentiate [weighted entropy loss, negative Q loss] w.r.t. actor params.

    The VJP shares samples/dropout across both terms. Norms are global, after
    batch averaging and before Adam; they are not norms of parameter updates.
    """
    terms, pullback, info = jax.vjp(loss_terms_fn, params, has_aux=True)
    entropy_grads, = pullback(jnp.array([1., 0.], dtype=terms.dtype))
    q_grads, = pullback(jnp.array([0., 1.], dtype=terms.dtype))
    grads = jax.tree.map(lambda h, q: h + q, entropy_grads, q_grads)
    h_norm = optax.global_norm(entropy_grads)
    q_norm = optax.global_norm(q_grads)
    dot = sum(jnp.sum(h * q) for h, q in zip(
        jax.tree.leaves(entropy_grads), jax.tree.leaves(q_grads)))
    eps = jnp.asarray(1e-12, dtype=h_norm.dtype)
    info = dict(info)
    info.update({
        'edit_entropy_grad_norm': h_norm,
        'edit_q_grad_norm': q_norm,
        'edit_entropy_to_q_grad_ratio': h_norm / jnp.maximum(q_norm, eps),
        'edit_grad_cosine': jnp.clip(dot / jnp.maximum(h_norm * q_norm, eps), -1., 1.),
        'edit_actor_grad_norm': optax.global_norm(grads),
    })
    return grads, info
