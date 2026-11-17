"""Bound readiness waits for large state trees sharing a GPU client with rollout."""
import jax


def block_until_ready_serial(tree):
    """Wait for every leaf without JAX's parallel batched readiness fan-out.

    A learner has thousands of state leaves. Submitting all readiness futures at
    once can monopolize the shared CUDA client's event/stream workers while the
    update is still running. Serial waits retain the same completion guarantee.
    """
    for leaf in jax.tree.leaves(tree):
        wait = getattr(leaf, 'block_until_ready', None)
        if wait is not None:
            wait()
    return tree
