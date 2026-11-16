"""Server-only rollout diagnostics; never changes the learner's train state."""
import json
import logging
from pathlib import Path
import time

import jax
import numpy as np
import yaml


def prepare_update(agent, processor, task, utd_ratio, replicated_sharding):
    """Compile the actual update signature, without executing or changing state."""
    if task.env_type != 'ur':
        return None
    settings = yaml.safe_load(Path(task.rollout_config_path).read_text())
    enabled = settings.get('warmup_update', True)
    if type(enabled) is not bool:
        raise ValueError('rollout warmup_update must be a YAML boolean')
    if not enabled:
        return None
    from expo_ft.data.batch_processor import BatchProcessor
    # Separate iterators: warmup must not consume the live prefetch sequence.
    buffers = (processor.replay_buffer, processor.offline_replay_buffer)
    previous = [(b, hasattr(b, 'rng'), getattr(b, 'rng', None)) for b in buffers]
    logging.info('Compiling learner update before robot connection (no parameter updates)...')
    start = time.perf_counter()
    try:
        probe = BatchProcessor(
            processor.replay_buffer, processor.offline_replay_buffer,
            processor.data_sharding, processor.batch_size, processor.utd_ratio,
            processor.offline_ratio, processor.actor_success_only,
            processor.use_dagger_hil_sampling,
        )
        batch, actor_batch, _ = probe.next_batch(jax.random.PRNGKey(0))
        clean = agent.replace(_infer_cache=None, rng=jax.device_put(agent.rng, replicated_sharding))
        # AOT lowering reconstructs TrainState with JAX ArgInfo placeholders;
        # OpenPI's runtime array checks don't accept those metadata objects.
        from openpi.shared import array_typing as at
        previous_check = at.config.jaxtyping_disable
        try:
            at.config.update('jaxtyping_disable', True)
            def signature(tree):
                leaves, structure = jax.tree.flatten(tree)
                return structure, tuple(
                    (jax.core.get_aval(x).shape, jax.core.get_aval(x).dtype,
                     jax.core.get_aval(x).weak_type, getattr(x, 'sharding', None))
                    for x in leaves)

            executables = []
            current = clean
            # E.g. the temperature and Adam moments lose weak typing over the
            # first two updates. Follow abstract outputs to a stable signature.
            for _ in range(4):
                logging.info('Compiling learner update signature %d...', len(executables) + 1)
                executable = type(clean)._update_jit.lower(
                    current, current, batch, utd_ratio, actor_batch).compile()
                executables.append(executable)
                abstract, _ = jax.eval_shape(
                    lambda a, b, c: type(a)._update_jit(a, a, b, utd_ratio, c),
                    current, batch, actor_batch)
                steady = jax.tree.map(
                    lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype,
                        weak_type=a.weak_type, sharding=s),
                    abstract, executable.output_shardings[0])
                steady = steady.replace(rng=jax.ShapeDtypeStruct(
                    steady.rng.shape, steady.rng.dtype, sharding=replicated_sharding))
                if signature(current) == signature(steady):
                    break
                current = steady
            else:
                raise RuntimeError('Update signature did not stabilize during compile-only warmup')

        finally:
            at.config.update('jaxtyping_disable', previous_check)
    finally:
        for b, existed, rng in previous:
            if existed:
                b.rng = rng
            elif hasattr(b, 'rng'):
                del b.rng
    logging.info('Learner update compilation finished in %.2fs; parameters, optimizer and sampling RNG unchanged.',
                 time.perf_counter() - start)
    return tuple(executables)


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
