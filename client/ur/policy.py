"""One evaluation interface for full/LoRA JAX SFT and complete EXPO checkpoints."""
import json
import time
from pathlib import Path

import numpy as np


def load_norm_stats(path):
    data = json.loads(Path(path).expanduser().read_text())
    return data.get('norm_stats', data)


def validate_config(config, *, files=True):
    from .utils import validate_single_arm_norm_stats
    model = config['model']
    if model['kind'] not in {'sft', 'expo'}:
        raise ValueError('model.kind must be sft or expo')
    if model['architecture'] not in {'full', 'lora'}:
        raise ValueError('model.architecture must explicitly select full or lora')
    if not 1 <= model['replan_steps'] <= model['action_horizon']:
        raise ValueError('Require 1 <= replan_steps <= action_horizon')
    if model['kind'] == 'expo' and model.get('sample_steps', 10) != 10:
        raise ValueError('Standard EXPO uses the upstream 10-step denoising configuration')
    if config['robot']['action_fps'] != 10:
        raise ValueError('This standard EXPO UR deployment uses 10 Hz actions')
    if config['robot']['force_mode']['rtde_freq'] != 100 or not config['robot']['force_mode']['continuous_tracking']:
        raise ValueError('Expected continuous 100 Hz UR force tracking')
    if files:
        checkpoint = Path(model['checkpoint_dir']).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        stats = load_norm_stats(model['norm_stats_path'])
        validate_single_arm_norm_stats(stats, fixed_gripper=False, learned_state_dim=19,
                                       path=model['norm_stats_path'], padded_action_dim=32)
        if model['kind'] == 'sft' and not (checkpoint / 'params').is_dir():
            raise FileNotFoundError(checkpoint / 'params')
        if model['kind'] == 'expo' and not Path(model['run_config']).expanduser().is_file():
            raise FileNotFoundError(model['run_config'])


def example_observation():
    return {'observation/state': np.zeros(19, np.float32),
            'observation/image': np.zeros((224, 224, 3), np.uint8),
            'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
            'prompt': 'Insert the black rack into the blue slot'}


class RemotePolicy:
    def __init__(self, host, port, expected=None):
        # websockets 13 uses different header keywords than later OpenPI clients.
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.connection = connect(f'ws://{host}:{port}', compression=None, max_size=None)
        self.metadata = msgpack_numpy.unpackb(self.connection.recv())
        if expected is not None:
            for key in ('kind', 'architecture', 'replan_steps'):
                if self.metadata.get(key) != expected[key]:
                    self.connection.close()
                    raise ValueError(f"Remote model metadata mismatch: {key}")

    def infer(self, obs):
        self.connection.send(self.packer.pack(obs))
        raw = self.connection.recv()
        if isinstance(raw, str):
            raise RuntimeError(raw)
        return self.codec.unpackb(raw)

    def close(self):
        self.connection.close()


class URPolicy:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        self.steps = config['model']['replan_steps']
        self.kind = config['model']['kind']
        if self.kind == 'sft':
            self._load_sft()
        else:
            self._load_expo()

    def _load_sft(self):
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.policies.policy import Policy
        from openpi.shared.normalize import NormStats
        from expo_ft.agents.vla.ur5e_data import make_train_config
        from expo_ft.utils.train_utils import _restore_pi05_params
        cfg = self.config['model']
        train = make_train_config(lora=cfg['architecture'] == 'lora', action_horizon=cfg['action_horizon'])
        params = _restore_pi05_params(str(Path(cfg['checkpoint_dir']).expanduser() / 'params'))
        # Strict evaluation restore: a wrong LoRA config must not silently drop adapters.
        model = train.model.load(jax.tree.map(jnp.asarray, params), remove_extra_params=False)
        model.eval()
        stats = {key: NormStats(**value) for key, value in load_norm_stats(cfg['norm_stats_path']).items()}
        dc = train.data.create(train.assets_dirs, train.model)
        self.policy = Policy(model, transforms=[*dc.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs], output_transforms=[*dc.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
            *dc.data_transforms.outputs], sample_kwargs={'num_steps': cfg.get('sample_steps', 10), 'num_samples': 1})

    def _load_expo(self):
        import jax
        from ml_collections import ConfigDict
        from openpi.training import sharding
        from expo_ft.agents.vla.pi05 import build_pi05
        from expo_ft.agents.alg.expo_ft import load_agent
        from expo_ft.agents.alg.batch_utils import CRITIC_CAMERA_KEYS
        from expo_ft.data.replay_buffer import create_replay_buffer, _critic_key_to_storage
        from client.envs.ur_env import to_expo_observation
        cfg = self.config['model']
        run = json.loads(Path(cfg['run_config']).expanduser().read_text())
        config = ConfigDict(run['agent_config'])
        if config.model_cls != 'EXPOLearner':
            raise ValueError('Only standard EXPOLearner is supported')
        expected = 'expo_pi05_ur_' + cfg['architecture']
        if config.pi05_config_name != expected or config.pi05_action_horizon != cfg['action_horizon'] or run['replan_steps'] != self.steps:
            raise ValueError('Evaluation architecture/horizon/replan_steps must match ur_run.json')
        # Relocation of the original statistics is allowed, but their content must match.
        import hashlib
        norm_path = Path(cfg['norm_stats_path']).expanduser()
        if hashlib.sha256(norm_path.read_bytes()).hexdigest() != run['norm_stats_sha256']:
            raise ValueError('Evaluation normalization does not match the online run')
        config.pi05_assets_dir = str(norm_path.parent)
        config.pi05_asset_id = '.'
        config.pi05_weight_loader_path = ''
        mesh = sharding.make_mesh(1)
        ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rs = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        actor, state, target, kwargs, metadata = build_pi05(config, 0, mesh, ds, rs,
            resume=True, default_prompt=self.config['task']['description'])
        keys = tuple(run.get('critic_camera_keys', CRITIC_CAMERA_KEYS))
        buffer = create_replay_buffer(config=config, example_action=np.zeros((1, 7)),
            capacity=cfg['action_horizon']+2, task_description=self.config['task']['description'],
            replan_steps=self.steps, seed=0, critic_camera_keys=keys)
        buffer.insert_dataset([dict(observations=to_expo_observation(example_observation()),
            actions=np.zeros(7), rewards=1., masks=0., dones=1.)])
        example = {_critic_key_to_storage(k): buffer.dataset_dict[_critic_key_to_storage(k)][0][None] for k in keys}
        for k in ['state', 'actions']:
            example[k] = buffer.dataset_dict[k][0][None]
        obs, st, act = buffer.convert_to_critic_format(example)
        actor.action_dim, actor.state_dim = act.squeeze().shape[-1], st.squeeze().shape[-1]
        kwargs['critic_camera_keys'] = keys
        self.agent = load_agent(seed=0, example_observation=obs.squeeze(), example_action=act.squeeze(),
            example_state=st.squeeze(), actor=actor, actor_train_state=state, target_actor_params=target,
            agent_kwargs=kwargs, metadata=metadata, mesh=mesh, data_sharding=ds,
            replicated_sharding=rs, resume=True, replan_steps=self.steps,
            default_prompt=self.config['task']['description'], edit_action_xyzg=False)
        self.agent = restore_online_agent(cfg['checkpoint_dir'], cfg.get('checkpoint_step'), self.agent)

    def infer(self, obs, **kwargs):
        from client.envs.ur_env import to_expo_observation
        started = time.perf_counter()
        raw = to_expo_observation(obs)
        if self.kind == 'sft':
            actions = self.policy.infer(raw, **kwargs)['actions']
        else:
            actions, self.agent, _ = self.agent.sample_actions(raw)
        actions = np.asarray(actions).reshape(-1, 7)[:self.steps]
        if actions.shape != (self.steps, 7) or not np.isfinite(actions).all():
            raise ValueError('Policy returned invalid UR action chunk')
        return {'actions': actions, 'controller': self.kind,
                'policy_timing': {'infer_ms': 1000 * (time.perf_counter() - started)}}


def create_policy(config, *, timing_recorder=None):
    if config['policy'].get('location') == 'remote':
        return RemotePolicy(config['policy']['host'], config['policy']['port'], config['model'])
    return URPolicy(config)


def warmup_policy(policy, obs, *, timing_recorder=None):
    policy.infer(obs)


def check_model(config):
    policy = URPolicy(config)
    result = policy.infer(example_observation())
    print('PASS model inference', result['actions'].shape, result['policy_timing'])


def restore_online_agent(checkpoint_dir, step, agent):
    """Read EXPO's agent + params checkpoint without creating or modifying a run."""
    import orbax.checkpoint as ocp
    from expo_ft.agents.alg.expo_ft import restore_checkpoint
    root = Path(checkpoint_dir).expanduser().resolve()
    if root.name.isdigit():
        step, root = int(root.name), root.parent
    manager = ocp.CheckpointManager(root, item_handlers={
        'agent': ocp.PyTreeCheckpointHandler(), 'params': ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(create=False, read_only=True))
    try:
        step = manager.latest_step() if step is None else int(step)
        if step is None:
            raise ValueError(f'No online checkpoint in {root}')
        return restore_checkpoint(manager, agent, step=step).cache_infer_params()
    finally:
        manager.close()
