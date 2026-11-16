"""Learner-only transition bookkeeping; run in the server root .venv."""
import os
import tempfile
import textwrap
from types import SimpleNamespace
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


class LearnerStartupTests(unittest.TestCase):
    def test_fresh_checkpoint_initialization_in_both_entrypoints(self):
        from etils import epath
        from expo_ft.agents import initialize_checkpoint_dir
        for filename in ['train_pi_robo.py', 'train_pi_robo_async.py']:
            with self.subTest(entrypoint=filename), tempfile.TemporaryDirectory() as root:
                source = Path(filename).read_text()
                # Execute the real startup block, stopping before W&B/model/robot initialization.
                block = source[source.index('    log_dir ='):source.index('    from expo_ft.utils.ur_run import')]
                flags = SimpleNamespace(output_dir=root, run_name='fresh', keep_period=None,
                                        overwrite=False, resume=False)
                ns = dict(os=os, FLAGS=flags, epath=epath,
                          initialize_checkpoint_dir=initialize_checkpoint_dir)
                exec(compile(textwrap.dedent(block), filename, 'exec'), ns)
                try:
                    self.assertFalse(ns['resuming'])
                    self.assertTrue(Path(ns['checkpoint_dir']).is_dir())
                finally:
                    ns['checkpoint_manager'].close()
                # Existing output must still require an explicit resume/overwrite decision.
                with self.assertRaises(FileExistsError):
                    exec(compile(textwrap.dedent(block), filename, 'exec'), ns)


class LearnerTransitionTests(unittest.TestCase):
    def test_skipped_terminal_step_does_not_steal_previous_action_label(self):
        from expo_ft.utils.loop_utils import EpisodeSink
        processor=Mock()
        sink=EpisodeSink(processor,threading.Lock(),None,Path('/tmp'),None,
            save_buffer=False,checkpoint_model=False,checkpoint_interval=10,start_step=0)
        action=np.arange(7)
        sink.record_transition(1,{'actions':action,'is_hil':True})
        sink.label_last(0.,1.,False)
        # The protocol reports executed=False, so the learner records no new transition.
        sink.label_last(1.,0.,True)
        sink.flush_transitions()
        processor.insert_transition.assert_called_once()
        recorded=processor.insert_transition.call_args.args[0]
        np.testing.assert_array_equal(recorded['actions'],action)
        self.assertEqual(recorded['rewards'],1.)
        self.assertTrue(recorded['dones'])

if __name__ == "__main__":
    unittest.main()
