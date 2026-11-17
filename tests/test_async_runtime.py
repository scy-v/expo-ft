import ast
from collections import deque
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class AsyncFailureTests(unittest.TestCase):
    def test_worker_failure_closes_robot_and_surfaces_to_rollout(self):
        main = next(n for n in ast.parse(Path('train_pi_robo_async.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        nodes = [n for n in main.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'_update_worker', '_raise_worker_error'}]
        failed = RuntimeError('simulated learner OOM')
        env = dict(FLAGS=SimpleNamespace(seed=42, ep_timeout_secs=0),
                   jax=Mock(), time=Mock(), deque=deque, agent=object(),
                   _can_update=threading.Event(), _stop_event=threading.Event(),
                   _episode_done=threading.Event(), _update_count=[0], _worker_error=[None],
                   _save_pending_checkpoint=Mock(), buffer_lock=threading.Lock(),
                   batch_processor=Mock(), env=SimpleNamespace(client=Mock()), logging=logging)
        env['_can_update'].set()
        env['batch_processor'].next_batch.side_effect = failed
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<worker>', 'exec'), env)
        env['_update_worker']()
        self.assertIs(env['_worker_error'][0], failed)
        self.assertTrue(env['_stop_event'].is_set())
        env['env'].client.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'stopping rollout') as caught:
            env['_raise_worker_error']()
        self.assertIs(caught.exception.__cause__, failed)

    def test_wrapper_does_not_recover_after_learner_failure(self):
        from expo_ft.env.env_client import EnvClientWrapper
        wrapper = EnvClientWrapper.__new__(EnvClientWrapper)
        wrapper.client = Mock()
        wrapper.abort_check = Mock(side_effect=[None, RuntimeError('learner failed')])
        with patch('expo_ft.env.env_client.EnvClient') as recreate:
            with self.assertRaisesRegex(RuntimeError, 'learner failed'):
                wrapper._call('step', Mock(side_effect=RuntimeError('disconnected')))
            recreate.assert_not_called()

    def test_inflight_rpc_does_not_wait_for_reconnection(self):
        from expo_ft.env.env_client import EnvClient
        client = EnvClient.__new__(EnvClient)
        client.abort_check = Mock(side_effect=[None, RuntimeError('learner failed')])
        client._packer = Mock()
        connection = Mock()
        connection.recv.side_effect = OSError('closed by learner')
        client._get_connection = Mock(return_value=connection)
        with self.assertRaisesRegex(RuntimeError, 'learner failed'):
            client._call_operation('reset', {})
        client._get_connection.assert_called_once()

    def test_accept_wait_can_be_aborted(self):
        from expo_ft.env.env_client import EnvClient
        client = EnvClient.__new__(EnvClient)
        client._ensure_server = Mock()
        client._cond = threading.Condition()
        client._pending = None
        client.abort_check = Mock(side_effect=RuntimeError('learner failed'))
        with self.assertRaisesRegex(RuntimeError, 'learner failed'):
            client._accept()


class AsyncLaunchTests(unittest.TestCase):
    def test_documented_launch_matches_script_and_gpu_layout(self):
        import re
        import shlex
        text = Path('expo-setup/训练命令.md').read_text()
        block = next(b for b in re.findall(r'```bash\n(.*?)```', text, re.S)
                     if 'python train_pi_robo_async.py' in b)
        args = shlex.split(block.replace('\\\n', ' '))
        self.assertIn('CUDA_VISIBLE_DEVICES=2,0,1', args)
        self.assertIn('--batch_size=64', args)
        self.assertFalse(any(a.startswith(('--num_updates', '--update_type')) for a in args))
        script = Path('scripts/ur/run_server_async.sh').read_text()
        self.assertIn('${CUDA_VISIBLE_DEVICES:-2,0,1}', script)
        self.assertIn('python train_pi_robo_async.py', script)
        self.assertNotIn('--num_updates', script)
        source = Path('train_pi_robo_async.py').read_text()
        import textwrap
        code = source[source.index('    num_gpus = jax.device_count()'):source.index('    mesh = jax.sharding.Mesh(')]
        scope = dict(jax=SimpleNamespace(device_count=lambda:3, devices=lambda:[2,0,1]),
                     FLAGS=SimpleNamespace(batch_size=64, fsdp_devices=1))
        exec(textwrap.dedent(code), scope)
        self.assertEqual(scope['sample_device'], 2)
        self.assertEqual(scope['update_devices'], [0,1])
