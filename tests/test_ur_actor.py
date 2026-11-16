"""Actor-only preflight and remote evaluation tests; no hardware or model weights."""
from contextlib import contextmanager
import copy
import os
import tempfile
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve
import yaml

from client.ur import check
from client.ur.policy import RemotePolicy, example_observation


@contextmanager
def policy_server(handler):
    failures = []
    def guarded(connection):
        try:
            handler(connection)
        except Exception as exc:
            failures.append(exc)
    server = serve(guarded, '127.0.0.1', 0, compression=None, max_size=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.socket.getsockname()[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        if thread.is_alive() or failures:
            raise AssertionError(f'Policy server did not finish cleanly: {failures}')


class RemotePolicyTests(unittest.TestCase):
    metadata = dict(kind='sft', architecture='full', replan_steps=8)

    def test_remote_observations_actions_and_server_error(self):
        actions = np.arange(56, dtype=np.float32).reshape(8, 7)
        packer = msgpack_numpy.Packer()
        def handler(connection):
            connection.send(packer.pack(self.metadata))
            observation = msgpack_numpy.unpackb(connection.recv())
            self.assertEqual(observation['observation/state'].shape, (19,))
            self.assertEqual(observation['observation/image'].shape, (224, 224, 3))
            connection.send(packer.pack(dict(actions=actions)))
            connection.recv()
            connection.send('synthetic inference failure')
        with policy_server(handler) as port:
            policy = RemotePolicy('127.0.0.1', port, self.metadata)
            try:
                np.testing.assert_array_equal(policy.infer(example_observation())['actions'], actions)
                with self.assertRaisesRegex(RuntimeError, 'synthetic inference failure'):
                    policy.infer(example_observation())
            finally:
                policy.close()

    def test_wrong_model_metadata_rejected_before_observation(self):
        from websockets.exceptions import ConnectionClosedOK
        def handler(connection):
            connection.send(msgpack_numpy.Packer().pack(self.metadata))
            with self.assertRaises(ConnectionClosedOK):
                connection.recv()
        with policy_server(handler) as port:
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                RemotePolicy('127.0.0.1', port, dict(self.metadata, kind='expo'))


class PreflightTests(unittest.TestCase):
    def test_remote_configs_do_not_require_local_weights(self):
        for path in ['configs/ur/insert_sft.yaml', 'configs/ur/insert_expo.yaml']:
            with patch('client.ur.policy.load_norm_stats', side_effect=AssertionError('local weights accessed')):
                self.assertEqual(check.check_config(path)['policy']['location'], 'remote')

    def test_training_and_evaluation_configs_are_separate(self):
        config = check.check_config('configs/ur/insert.yaml')
        self.assertEqual(config['online']['max_episode_chunks'], 60)
        for key in ('model', 'policy'):
            self.assertNotIn(key, config)
        for key in ('num_episodes', 'episode_timeout_sec'):
            self.assertNotIn(key, config['record'])
        for name in ('insert_sft.yaml', 'insert_expo.yaml'):
            evaluation = check.check_config(Path('configs/ur') / name)
            self.assertNotIn('online', evaluation)
            self.assertEqual(evaluation['record']['num_episodes'], 20)

    def test_invalid_config_and_missing_camera_are_reported_as_failures(self):
        config = yaml.safe_load(Path('configs/ur/insert.yaml').read_text())
        changed = copy.deepcopy(config)
        changed['robot']['action_fps'] = 20
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'invalid.yaml'
            path.write_text(yaml.safe_dump(changed))
            results = []
            check.record(results, 'invalid', lambda: check.check_config(path))
        self.assertEqual(results[0]['status'], 'fail')
        with patch.object(check, 'camera_inventory', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'Missing cameras'):
                check.check_cameras(config['cameras'])

    def test_serial_check_does_not_open_device(self):
        with tempfile.NamedTemporaryFile() as stream:
            with patch('builtins.open', side_effect=AssertionError('opened serial')):
                self.assertFalse(check.check_serial(stream.name)['opened'])


class LogPathTests(unittest.TestCase):
    def test_latest_link_resolves_relative_output_path(self):
        from client.ur.ur import update_latest_symlink
        with tempfile.TemporaryDirectory(dir='.') as root:
            target = Path(os.path.relpath(root)) / 'all_logs' / 'run.yaml'
            target.parent.mkdir()
            target.write_text('test: true\n')
            link = target.parent.parent / 'latest.yaml'
            update_latest_symlink(target, link)
            self.assertEqual(link.read_text(), target.read_text())


class RecorderLifecycleTests(unittest.TestCase):
    def test_evaluation_closes_recorder_after_normal_and_failed_run(self):
        from client.ur.ur import Inference
        for error in (None, RuntimeError('evaluation failed'), KeyboardInterrupt()):
            with self.subTest(error=error):
                engine = Inference.__new__(Inference)
                engine.recorder = Mock()
                def evaluate():
                    engine.recorder.close.assert_not_called()
                    if error is not None:
                        raise error
                engine._run_evaluation = evaluate
                if error is None:
                    engine.run()
                else:
                    with self.assertRaises(type(error)):
                        engine.run()
                engine.recorder.close.assert_called_once()

    def test_online_initialization_uses_hardware_config_without_local_weights(self):
        from contextlib import ExitStack
        from client.envs.ur_env import UREnv
        from client.ur.online_backend import OnlineInference
        config = yaml.safe_load(Path('configs/ur/insert.yaml').read_text())
        with tempfile.TemporaryDirectory() as root, ExitStack() as patches:
            config['output_root'] = str(Path(root) / 'output')
            config['video']['visualize'] = False
            path = Path(root) / 'actor.yaml'
            path.write_text(yaml.safe_dump(config))
            for method in ['connect_robot', 'connect_cameras', 'connect_gripper', 'wait_for_gripper_states']:
                patches.enter_context(patch.object(OnlineInference, method))
            patches.enter_context(patch('client.ur.keyboard_listener.KeyboardListener'))
            patches.enter_context(patch('client.ur.robot_lifecycle.register_robot_cleanup'))
            env = UREnv(ur_config_path=path, replan_steps=8)
            try:
                self.assertTrue(env.engine.remote_policy)
                self.assertIsNone(env.engine.norm_stats)
                self.assertEqual(env.engine.close_threshold, 0.5)
                self.assertEqual(env.engine.action_fps, 10)
                self.assertEqual(env.engine.rtde_freq, 100)
                self.assertFalse(env.engine.recorder.retain_frames)
            finally:
                env.close()
            self.assertFalse(env.engine.recorder._visualizer.is_alive())
            self.assertFalse(env.engine.recorder._logger.is_alive())

    def test_online_close_drains_logs_and_stops_threads(self):
        from client.ur.recording import Recorder
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            recorder = Recorder(path / 'actions.yaml', [path / 'wrist.mp4', path / 'ext.mp4'], visualize=False)
            try:
                recorder.submit_actions(np.ones((1, 7)), 1, 'insert')
                recorder.submit_obs(example_observation())
            finally:
                recorder.close()
            recorder.close()
            self.assertFalse(recorder._logger.is_alive())
            self.assertFalse(recorder._visualizer.is_alive())
            self.assertIn('infer_index: 1', (path / 'actions.yaml').read_text())


class VideoTests(unittest.TestCase):
    def test_mp4_encoder_roundtrip(self):
        from client.ur.recording import Recorder
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rgb.mp4'
            recorder = Recorder.__new__(Recorder)
            recorder.display_fps = 15
            recorder._encode([np.full((48, 64, 3), 100, np.uint8)], path)
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].to_ndarray(format='rgb24').shape, (48, 64, 3))


if __name__ == '__main__':
    unittest.main()
