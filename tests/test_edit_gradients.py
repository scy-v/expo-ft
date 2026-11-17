import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from expo_ft.utils.edit_gradients import edit_component_gradients


class EditGradientTests(unittest.TestCase):
    def test_jitted_tanh_policy_matches_original_gradient_and_adam_step(self):
        params = {'mu': jnp.array([.2, -.3]), 'log_std': jnp.array([-.4, -.6])}
        noise = jax.random.normal(jax.random.PRNGKey(7), (8, 2))

        def objectives(p, correction=True):
            z = p['mu'] + jnp.exp(p['log_std']) * noise
            u = jnp.tanh(z)
            log_p = (-.5 * noise**2 - p['log_std'] - .5 * jnp.log(2 * jnp.pi)
                     - jnp.log1p(-u**2)).sum(-1)
            if correction:
                log_p -= 2 * jnp.log(.2)
            q = -jnp.square(.2 * u - jnp.array([.15, -.1])).sum(-1)
            return jnp.stack([1.015 * log_p.mean(), -q.mean()])

        def loss_fn(p):
            terms = objectives(p)
            return terms, {'edit_actor_loss': terms.sum()}

        gradients, info = jax.jit(edit_component_gradients, static_argnums=0)(loss_fn, params)
        expected = jax.grad(lambda p: objectives(p).sum())(params)
        entropy = jax.grad(lambda p: objectives(p)[0])(params)
        q = jax.grad(lambda p: objectives(p)[1])(params)
        for a, b in zip(jax.tree.leaves(gradients), jax.tree.leaves(expected)):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(info['edit_entropy_grad_norm'], optax.global_norm(entropy), rtol=2e-6)
        np.testing.assert_allclose(info['edit_q_grad_norm'], optax.global_norm(q), rtol=2e-6)
        np.testing.assert_allclose(info['edit_entropy_to_q_grad_ratio'],
                                   optax.global_norm(entropy) / optax.global_norm(q), rtol=2e-6)
        # The scaling density correction changes loss values, not actor gradients.
        no_correction = jax.grad(lambda p: objectives(p, False)[0])(params)
        for a, b in zip(jax.tree.leaves(entropy), jax.tree.leaves(no_correction)):
            np.testing.assert_allclose(a, b, atol=1e-7)
        optimizer = optax.adam(3e-4)
        state = optimizer.init(params)
        actual_updates, _ = optimizer.update(gradients, state, params)
        expected_updates, _ = optimizer.update(expected, state, params)
        for a, b in zip(jax.tree.leaves(actual_updates), jax.tree.leaves(expected_updates)):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-8)

    def test_opposed_gradients_and_zero_q_are_finite(self):
        def opposed(p):
            return jnp.stack([p.sum(), -p.sum()]), {}
        grads, info = edit_component_gradients(opposed, jnp.ones(4))
        np.testing.assert_allclose(grads, 0)
        self.assertAlmostEqual(float(info['edit_grad_cosine']), -1.)
        self.assertAlmostEqual(float(info['edit_entropy_to_q_grad_ratio']), 1.)

        def zero_q(p):
            return jnp.stack([p.sum(), jnp.asarray(0.)]), {}
        _, info = edit_component_gradients(zero_q, jnp.ones(4))
        self.assertEqual(float(info['edit_q_grad_norm']), 0.)
        self.assertTrue(all(np.isfinite(float(v)) for v in info.values()))

    def test_forward_is_shared_between_gradient_components(self):
        calls = []
        def loss_fn(p):
            calls.append(True)
            return jnp.stack([p.sum(), (p**2).sum()]), {}
        edit_component_gradients(loss_fn, jnp.ones(2))
        self.assertEqual(len(calls), 1)
