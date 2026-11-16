"""Learner-only transition bookkeeping; run in the server root .venv."""
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


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
