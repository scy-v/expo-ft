import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from client.ur.keyboard import CheckpointKeys


class CheckpointKeyTests(unittest.TestCase):
    def test_edges_window_focus_and_reset(self):
        keys = CheckpointKeys()
        self.assertFalse(keys.consume([('Key.minus', 1)]))
        self.assertFalse(keys.consume([('Key.minus', 1.6)]))
        self.assertTrue(keys.consume([('Key.minus', 2.1)]))
        keys.reset()
        self.assertFalse(keys.consume([('Key.minus', 3), ('Key.focus_lost', 3.1), ('Key.minus', 3.2)]))
        self.assertTrue(keys.consume([('Key.minus', 3.4)]))
        keys.reset()
        self.assertFalse(keys.requested)
        self.assertFalse(keys.consume([('Key.space', 4), ('Key.space', 4.1)]))


class CheckpointSaveTests(unittest.TestCase):
    def make_sink(self):
        from expo_ft.utils.loop_utils import EpisodeSink
        save = Mock()
        sink = EpisodeSink(Mock(), threading.Lock(), Mock(), Path('/tmp'), save,
                           save_buffer=False, checkpoint_model=True,
                           checkpoint_interval=2000, start_step=0)
        return sink, save

    def test_keyboard_supplements_interval_without_shifting_it(self):
        sink, save = self.make_sink()
        sink.set_checkpoint_control(dict(manual=True, requested=False))
        with patch('sys.stdin.isatty', return_value=False):
            sink.flush_episode(1999, object())
            save.assert_not_called()
            sink.set_checkpoint_control(dict(manual=True, requested=True))
            sink.flush_episode(1999, object())
            self.assertEqual(save.call_count, 1)
            self.assertEqual(sink.next_interval_step, 2000)
            sink.flush_episode(2001, object())
            self.assertEqual(save.call_count, 2)
            self.assertEqual(sink.next_interval_step, 4000)
            sink.set_checkpoint_control(dict(manual=True, requested=True))
            sink.flush_episode(4002, object())
            self.assertEqual(save.call_count, 3)
            sink.set_checkpoint_control(dict(manual=True, requested=True))
            sink.flush_episode(4002, object())
            self.assertEqual(save.call_count, 3)
            sink.flush_episode(4003, object())
            self.assertEqual(save.call_count, 3)

    def test_failed_save_retains_request_and_interval_for_retry(self):
        sink, save = self.make_sink()
        sink.set_checkpoint_control(dict(manual=True, requested=True))
        save.side_effect = RuntimeError('disk failed')
        with patch('sys.stdin.isatty', return_value=False):
            sink.flush_episode(2100, object())
            self.assertTrue(sink.checkpoint_requested)
            self.assertEqual(sink.next_interval_step, 2000)
            self.assertIsNone(sink.last_saved_step)
            save.side_effect = None
            sink.flush_episode(2100, object())
            self.assertFalse(sink.checkpoint_requested)
            self.assertEqual(sink.next_interval_step, 4000)

    def test_author_interval_mode(self):
        sink, save = self.make_sink()
        sink.set_checkpoint_control(dict(manual=False, requested=True))
        with patch('sys.stdin.isatty', return_value=False):
            sink.maybe_save_checkpoint(1999, object())
            save.assert_not_called()
            sink.maybe_save_checkpoint(2405, object())
            save.assert_called_once()

    def test_metadata_roundtrip_without_touching_observation(self):
        from expo_ft.env.env_client import EnvClient, EnvClientWrapper
        client = EnvClient.__new__(EnvClient)
        observation = {'state': [1, 2]}
        client._call_operation = Mock(return_value=dict(observation=observation,
            checkpoint_control=dict(manual=True, requested=True), done=True, success=True, reward=1., mask=0.))
        self.assertIs(client.get_observation('ur'), observation)
        wrapper = EnvClientWrapper.__new__(EnvClientWrapper)
        wrapper.client = client
        self.assertEqual(wrapper.checkpoint_control, dict(manual=True, requested=True))
        self.assertEqual(client.get_info_for_step('ur'), (True, True, 1., 0.))


class AsyncCheckpointTests(unittest.TestCase):
    """Execute the production callbacks without importing the training CLI flags."""
    def callbacks(self):
        import ast
        import logging
        main = next(n for n in ast.parse(Path('train_pi_robo_async.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        names = {'_save_from_update_thread', '_save_pending_checkpoint'}
        nodes = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = dict(threading=threading, logging=logging, agent=object(),
                   checkpoint_manager=Mock(), save_checkpoint=Mock(),
                   _can_update=threading.Event(), _ckpt_done=threading.Event(),
                   _ckpt_request=[None], _ckpt_error=[None], _update_thread=Mock())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<async-checkpoint>', 'exec'), env)
        return env

    def test_before_learning_saves_initial_full_agent(self):
        env = self.callbacks()
        env['_save_from_update_thread'](None, object(), 2000)
        env['save_checkpoint'].assert_called_once_with(env['checkpoint_manager'], env['agent'], 2000)

    def test_requests_save_latest_learner_and_propagate_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                env = self.callbacks()
                env['_can_update'].set()
                learner = object()
                if failed:
                    env['save_checkpoint'].side_effect = OSError('disk failed')
                # The request callback blocks until the worker acknowledges its save.
                def respond(timeout):
                    self.assertEqual(env['_ckpt_request'][0], 2000)
                    env['_save_pending_checkpoint'](learner)
                    return env['_ckpt_done'].is_set()
                with patch.object(env['_ckpt_done'], 'wait', side_effect=respond):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, 'save failed'):
                            env['_save_from_update_thread'](None, object(), 2000)
                    else:
                        env['_save_from_update_thread'](None, object(), 2000)
                env['save_checkpoint'].assert_called_once_with(env['checkpoint_manager'], learner, 2000)
                self.assertIsNone(env['_ckpt_request'][0])

    def test_dead_update_thread_fails_without_reporting_saved(self):
        env = self.callbacks()
        env['_can_update'].set()
        env['_update_thread'].is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'not running'):
            env['_save_from_update_thread'](None, object(), 2000)
        env['save_checkpoint'].assert_not_called()

    def test_paused_worker_services_save_without_another_update(self):
        import ast
        from collections import deque
        from types import SimpleNamespace
        env = self.callbacks()
        env.update(FLAGS=SimpleNamespace(seed=42, ep_timeout_secs=120),
                   deque=deque, jax=Mock(), time=Mock(), sink=Mock(),
                   _stop_event=threading.Event(), _episode_done=Mock(),
                   _env_step=[2000], _update_count=[0])
        env['_can_update'].set()
        env['time'].time.side_effect = [0, 121]
        env['_episode_done'].wait.return_value = False
        calls = []
        def wait(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                env['_ckpt_request'][0] = 2000
            else:
                env['_stop_event'].set()
            return False
        env['_episode_done'].wait.side_effect = wait
        main = next(n for n in ast.parse(Path('train_pi_robo_async.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        worker = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == '_update_worker')
        exec(compile(ast.Module(body=[worker], type_ignores=[]), '<async-worker>', 'exec'), env)
        env['_update_worker']()
        env['save_checkpoint'].assert_called_once_with(env['checkpoint_manager'], env['agent'], 2000)
        self.assertEqual(env['_update_count'][0], 0)
        self.assertTrue(env['_ckpt_done'].is_set())
