"""Server-only rollout diagnostics; never changes the learner's train state."""
import json
import logging
from pathlib import Path
import time

import jax
import numpy as np
import yaml


def prepare_rollout(agent, config, task, replan_steps, delay, output_dir):
    if task.env_type != 'ur':
        return None
    settings = yaml.safe_load(Path(task.rollout_config_path).read_text())
    for key in ('warmup', 'sft_only'):
        if type(settings.get(key)) is not bool:
            raise ValueError(f'rollout {key} must be a YAML boolean')
    if settings['sft_only'] and delay != 0:
        raise ValueError('Fixed SFT diagnostic rollout requires standard delay=0')
    from client.ur.policy import URPolicy, example_observation
    from client.envs.ur_env import to_expo_observation
    fixed = None
    if settings['sft_only']:
        hardware = yaml.safe_load(Path(task.ur_config_path).read_text())
        hardware['model'] = dict(kind='sft', architecture='full',
            checkpoint_dir=str(Path(config.pi05_weight_loader_path).parent),
            norm_stats_path=str(Path(config.pi05_assets_dir) / config.pi05_asset_id / 'norm_stats.json'),
            action_horizon=config.pi05_action_horizon, replan_steps=replan_steps, sample_steps=10)
        # Allocate the separate, immutable full-SFT policy on the normal inference GPU.
        with jax.default_device(jax.devices()[0]):
            fixed = URPolicy(hardware)
        logging.info('DIAGNOSTIC rollout: fixed full SFT only; EXPO learner still updates normally.')
        settings['model'] = hardware['model']
    Path(output_dir, 'rollout_config.json').write_text(json.dumps(settings, indent=2)+'\n')
    if settings['warmup']:
        obs = example_observation()
        obs['prompt'] = task.language_instruction
        # EXPO rejects normalized-looking images (max <= 1); use raw uint8 pixels.
        for key in ('observation/image', 'observation/wrist_image'):
            obs[key] = np.full((224, 224, 3), 127, dtype=np.uint8)
        logging.info('Warming up rollout before robot connection (no robot actions)...')
        start = time.perf_counter()
        if fixed is None:
            actions, _, _ = agent.sample_actions(to_expo_observation(obs))
        else:
            with jax.default_device(jax.devices()[0]):
                actions = fixed.infer(obs)['actions']
        actions = np.asarray(jax.device_get(actions))
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('Invalid rollout warmup output')
        logging.info('Rollout warmup finished in %.2fs; discarded actions, learner unchanged.', time.perf_counter()-start)
    return fixed


def sample_fixed_sft(policy, observation):
    """Convert the existing online observation, bypassing all EXPO action selection."""
    obs = {'observation/state': observation['state'],
           'observation/image': observation['exterior_image_1_left'],
           'observation/wrist_image': observation['wrist_image_left'],
           'prompt': observation['prompt']}
    start = time.perf_counter()
    with jax.default_device(jax.devices()[0]):
        actions = policy.infer(obs)['actions']
    return actions, {'sample_time': time.perf_counter()-start, 'selected_action_type': 'main'}
