import threading
import io
import unittest
from unittest.mock import patch

from expo_ft.utils.update_progress import UpdateProgress, BAR_FORMAT


class UpdateProgressTests(unittest.TestCase):
    def test_elapsed_and_count_only_advance_on_completed_update(self):
        progress = UpdateProgress()
        self.assertEqual(progress.snapshot()['learner'], 'waiting_buffer')
        progress.stage('preparing_batch')
        self.assertEqual(progress.snapshot()['update_s'], '-')
        with patch('expo_ft.utils.update_progress.time.monotonic', return_value=10):
            progress.begin()
        with patch('expo_ft.utils.update_progress.time.monotonic', return_value=12.5):
            self.assertEqual(progress.snapshot()['update_s'], '2.5')
            self.assertEqual(progress.snapshot()['net_updates'], 0)
            progress.complete(1)
        progress.stage('paused')
        self.assertEqual(progress.snapshot(), dict(net_updates=1, learner='paused', update_s='-', last_s='2.5'))

    def test_refresh_runs_without_rollout_steps_and_stops(self):
        refreshed = threading.Event()

        class Bar:
            disable = False

            def set_postfix_str(self, values, refresh):
                refreshed.set()

        progress = UpdateProgress()
        progress.start(Bar())
        try:
            self.assertTrue(refreshed.wait(2))
        finally:
            progress.close()
        self.assertFalse(progress._thread.is_alive())

    def test_disabled_bar_starts_no_thread(self):
        progress = UpdateProgress()
        progress.start(type('Bar', (), {'disable': True})())
        self.assertIsNone(progress._thread)
        progress.close()

    def test_render_fits_narrow_terminal(self):
        from tqdm import tqdm
        for width in (40, 60, 80, 120):
            stream = io.StringIO()
            with tqdm(total=100001, initial=957, file=stream, ncols=width,
                      bar_format=BAR_FORMAT) as bar:
                bar.set_postfix_str('updates=12 run=23.5s last=57.2s')
                self.assertLessEqual(len(str(bar)), width)
                self.assertIn('updates=12', str(bar))
