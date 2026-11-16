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

    def test_manual_only_at_episode_end_once(self):
        sink, save = self.make_sink()
        sink.set_checkpoint_control(dict(manual=True, requested=False))
        with patch('sys.stdin.isatty', return_value=True), patch('select.select', side_effect=AssertionError('manual mode read stdin')):
            sink.maybe_save_checkpoint(9000, object())
            save.assert_not_called()
            sink.set_checkpoint_control(dict(manual=True, requested=True))
            save.assert_not_called()
            sink.flush_episode(9001, object())
            save.assert_called_once()
            sink.flush_episode(9002, object())
            save.assert_called_once()

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
