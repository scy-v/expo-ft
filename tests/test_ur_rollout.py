import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml


class RolloutTests(unittest.TestCase):
    def test_fixed_policy_bypasses_agent_and_keeps_latest_learner(self):
        from expo_ft.utils.loop_utils import AsyncChunkSampler
        from client.ur.policy import example_observation
        from client.envs.ur_env import to_expo_observation
        policy = Mock()
        policy.infer.return_value = {'actions': np.zeros((8, 7))}
        agent = Mock()
        sampler = AsyncChunkSampler(agent, delay=0, replan_steps=8, fixed_policy=policy)
        try:
            for current in (agent, Mock()):
                actions, returned, info = sampler.sample(current, to_expo_observation(example_observation()), {})
                self.assertIs(returned, current)
                current.sample_actions.assert_not_called()
                self.assertEqual(actions.shape, (8, 7))
            self.assertEqual(policy.infer.call_count, 2)
        finally:
            sampler.close()

    def test_warmup_discards_actions_before_environment_connection(self):
        from expo_ft.utils.ur_rollout import prepare_rollout
        agent = Mock()
        agent.sample_actions.return_value = (np.zeros((50, 7)), Mock(), {})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, 'rollout.yaml')
            path.write_text('warmup: true\nsft_only: false\n')
            task = SimpleNamespace(env_type='ur', rollout_config_path=path, language_instruction='test')
            self.assertIsNone(prepare_rollout(agent, None, task, 8, 0, root))
            agent.sample_actions.assert_called_once()
            raw = agent.sample_actions.call_args.args[0]
            self.assertEqual(raw['prompt'], 'test')
            self.assertEqual(raw['exterior_image_1_left'].dtype, np.uint8)
            self.assertGreater(raw['exterior_image_1_left'].max(), 1)
        for entry in ('train_pi_robo.py', 'train_pi_robo_async.py'):
            source = Path(entry).read_text()
            self.assertLess(source.index('fixed_rollout_policy = prepare_rollout'), source.index('    env = EnvClientWrapper'))

    def test_progress_uses_executed_action_budget(self):
        from client.ur.progress import EpisodeProgress
        stream = io.StringIO()
        stream.isatty = lambda: True
        progress = EpisodeProgress(60, 8, stream)
        self.assertIn('Chunks 1/60 +1/8', progress.update(1, 9))
        self.assertIn('Chunks 60/60 +0/8', progress.update(1, 480))
        progress.close()
        self.assertIn('\r', stream.getvalue())
        self.assertTrue(stream.getvalue().endswith('\n'))
