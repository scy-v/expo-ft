import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import expo_ft.agents
from expo_ft.data.replay_buffer import PiReplayBuffer, ALL_CAMERA_KEYS, _critic_key_to_storage
from expo_ft.utils.ur_rollout import prepare_update


def buffer():
    b = PiReplayBuffer.__new__(PiReplayBuffer)
    b._size = b._capacity = 32
    b._insert_index = 0
    b._replan_steps = 8
    b._delay = 0
    b._discount = .99
    b._valids_keep_terminal_windows = False
    b._seed = 42
    b._stage = {}
    b._gather_camera_keys = tuple(ALL_CAMERA_KEYS)
    b._gather_storage = tuple(map(_critic_key_to_storage, ALL_CAMERA_KEYS))
    b.dataset_dict = {k: np.full((32, 4, 4, 3), 127, np.uint8) for k in b._gather_storage}
    b.dataset_dict.update({k+'_mask': np.ones(32, bool) for k in b._gather_storage})
    b.dataset_dict.update(state=np.arange(1024, dtype=np.float32).reshape(32,32),
        actions=np.zeros((32,50,32), np.float32), rewards=np.zeros(32,np.float32),
        masks=np.ones(32,np.float32), dones=np.zeros(32,bool))
    return b


@struct.dataclass
class TinyLearner:
    params: object
    rng: object
    _infer_cache: object = None

    @partial(jax.jit, static_argnames='utd_ratio')
    def _update_jit(self, agent, batch, utd_ratio, actor_batch):
        return agent.replace(params=agent.params + batch['x'].sum() * utd_ratio), {}


class PrefetchTests(TestCase):
    def test_host_samples_match_old_values_and_own_staging(self):
        b = buffer()
        b.rng = jax.random.PRNGKey(42)
        original_rng = b.rng
        old = b._convert_to_openpi_format(b.sample_jax(16))
        b.rng = original_rng
        new = b._convert_to_openpi_format(b.sample_jax(16, host=True))
        for x, y in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
            self.assertIsInstance(y, np.ndarray)
            np.testing.assert_allclose(x, y, atol=1e-7)
        cam = ALL_CAMERA_KEYS[0]
        self.assertIs(new['next_image'][cam], new['next_delayed_image'][cam])
        saved = new['image'][cam].copy()
        for x in b.dataset_dict.values():
            x[...] = 0
        b.sample_jax(16, host=True)
        np.testing.assert_array_equal(saved, new['image'][cam])

    def test_iterator_keeps_pending_batches_on_host(self):
        b = buffer()
        iterator = b.get_iterator(sample_args={'batch_size': 16})
        batch = next(iterator)
        for item in [batch, *iterator.gi_frame.f_locals['queue']]:
            self.assertTrue(all(isinstance(x, np.ndarray) for x in jax.tree.leaves(item)))
        ds = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        sharded = b.apply_data_sharding(batch, ds)
        self.assertTrue(all(isinstance(x, jax.Array) for x in jax.tree.leaves(sharded)))

    def test_mixed_batches_stay_on_host_with_same_shuffle(self):
        from expo_ft.utils.train_utils import combine_batches
        a = {'x': np.arange(24).reshape(8,3)}
        b = {'x': np.arange(24,48).reshape(8,3)}
        key = jax.random.PRNGKey(7)
        host = combine_batches(a, b, key)
        device = combine_batches(jax.tree.map(jnp.asarray,a),jax.tree.map(jnp.asarray,b),key)
        self.assertIsInstance(host['x'], np.ndarray)
        np.testing.assert_array_equal(host['x'],device['x'])

    def test_compile_only_restores_rng_and_does_not_apply_updates(self):
        a, b = buffer(), buffer()
        a.rng = jax.random.PRNGKey(11)
        old_rng = a.rng
        processor = SimpleNamespace(replay_buffer=a, offline_replay_buffer=b,
            data_sharding=None,batch_size=64,utd_ratio=20,offline_ratio=0,
            actor_success_only=True,use_dagger_hil_sampling=False)
        def sample(_):
            a.rng = jax.random.PRNGKey(99)
            b.rng = jax.random.PRNGKey(99)
            return {'x': jnp.ones(4)}, None, None
        ds = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        agent = TinyLearner(jnp.array(1.), jax.random.PRNGKey(42))
        with tempfile.TemporaryDirectory() as root, patch('expo_ft.data.batch_processor.BatchProcessor') as factory:
            path = Path(root,'rollout.yaml')
            path.write_text('warmup_update: true\n')
            task = SimpleNamespace(env_type='ur',rollout_config_path=path)
            factory.return_value.next_batch.side_effect = sample
            executable = prepare_update(agent,processor,task,20,ds)
            self.assertIsNotNone(executable)
            self.assertEqual(float(agent.params),1.)
            np.testing.assert_array_equal(a.rng,old_rng)
            self.assertFalse(hasattr(b,'rng'))
            path.write_text('warmup_update: false\n')
            factory.reset_mock()
            self.assertIsNone(prepare_update(agent,processor,task,20,ds))
            factory.assert_not_called()

    def test_compile_failure_restores_checks_and_sampling_rng(self):
        from openpi.shared import array_typing as at
        a, b = buffer(), buffer()
        processor = SimpleNamespace(replay_buffer=a, offline_replay_buffer=b,
            data_sharding=None,batch_size=64,utd_ratio=20,offline_ratio=0,
            actor_success_only=True,use_dagger_hil_sampling=False)
        ds = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        agent = TinyLearner(jnp.array(1.),jax.random.PRNGKey(42))
        previous = at.config.jaxtyping_disable
        def sample(_):
            a.rng = jax.random.PRNGKey(99)
            return {'wrong_key': jnp.ones(4)}, None, None
        with tempfile.TemporaryDirectory() as root, patch('expo_ft.data.batch_processor.BatchProcessor') as factory:
            path = Path(root,'rollout.yaml')
            path.write_text('warmup_update: true\n')
            task = SimpleNamespace(env_type='ur',rollout_config_path=path)
            factory.return_value.next_batch.side_effect = sample
            with self.assertRaises(KeyError):
                prepare_update(agent,processor,task,20,ds)
        self.assertEqual(at.config.jaxtyping_disable,previous)
        self.assertFalse(hasattr(a,'rng'))
        for entry in ('train_pi_robo.py','train_pi_robo_async.py'):
            source = Path(entry).read_text()
            self.assertLess(source.index('update_executable = prepare_update'),source.index('    env = EnvClientWrapper'))

    def test_background_lookahead_does_not_upload_until_consumed(self):
        from expo_ft.data.batch_processor import BatchProcessor
        ds = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        bp = BatchProcessor(buffer(),buffer(),ds,16,1,0,False,False)
        batch, actor, _ = bp.next_batch(jax.random.PRNGKey(0),to_device=False)
        self.assertIsNone(actor)
        self.assertTrue(all(isinstance(x,np.ndarray) for x in jax.tree.leaves(batch)))
        batch, actor = bp.to_device(batch,actor)
        self.assertTrue(all(isinstance(x,jax.Array) for x in jax.tree.leaves(batch)))
