"""Actual Orbax roundtrip through the EXPO evaluation loader, on tiny CPU trees."""
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import struct
import orbax.checkpoint as ocp

from expo_ft.agents.alg.expo_ft import save_checkpoint
from client.ur.policy import restore_online_agent


@struct.dataclass
class State:
    params: object
    ema_params: object = None


@struct.dataclass
class Agent:
    batch_encoder: State
    edit_actor: State
    temp: State
    critic: State
    target_critic: State
    actor_train_state: State
    target_actor_params: object
    def cache_infer_params(self):return self


def agent(fill):
    states=[State({'w':jnp.full((2,3),fill+i,dtype=jnp.float32)}) for i in range(6)]
    return Agent(*states, {'target':jnp.full((3,),fill+6,dtype=jnp.float32)})


class CheckpointTests(unittest.TestCase):
    def test_native_checkpoint_preserves_actor_edit_critic_and_targets(self):
        with tempfile.TemporaryDirectory() as root:
            manager=ocp.CheckpointManager(root,item_handlers={
                'agent':ocp.PyTreeCheckpointHandler(),'params':ocp.PyTreeCheckpointHandler()},
                options=ocp.CheckpointManagerOptions(create=True))
            original=agent(10)
            save_checkpoint(manager,original,7)
            manager.wait_until_finished()
            manager.close()
            before=sorted(str(p.relative_to(root)) for p in Path(root).rglob('*'))
            restored=restore_online_agent(str(Path(root)/'7'),None,agent(0))
            for name in ['actor_train_state','edit_actor','critic','batch_encoder','temp','target_critic']:
                np.testing.assert_array_equal(getattr(restored,name).params['w'],getattr(original,name).params['w'])
            np.testing.assert_array_equal(restored.target_actor_params['target'],original.target_actor_params['target'])
            self.assertEqual(before,sorted(str(p.relative_to(root)) for p in Path(root).rglob('*')))


if __name__=='__main__':unittest.main()
