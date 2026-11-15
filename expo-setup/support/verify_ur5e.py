"""Audit converted UR5e data, replay transforms and an existing SFT checkpoint.

No robot connection, optimizer steps, or checkpoint writes are performed.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.dataset as pads
from flax import nnx, traverse_util
from openpi.models import model as model_lib, pi0_config
from openpi.training import config as oc
from openpi import transforms

from expo_ft.agents.vla.ur5e_data import UR5eDataConfig
from expo_ft.data.replay_buffer import PiReplayBuffer
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.train_utils import _restore_pi05_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Numeric OpenPI checkpoint directory, containing params and assets')
    parser.add_argument('--asset-id', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--action-horizon', type=int, default=50, help='Must match intended model configuration; checkpoint parameters alone do not encode this')
    parser.add_argument('--forward', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.converted / 'conversion.json').read_text())
    mapping = manifest['mapping']
    dataset = process_droid_dataset(str(args.converted), SimpleNamespace(action_space=mapping['action_space'], gripper_action_space=mapping['gripper_action_space']))
    source = pads.dataset(str(Path(manifest['source']) / 'data'), format='parquet')
    position = 0
    for ep in sorted(manifest['episodes'], key=lambda x: x['episode_index']):
        rows = source.to_table(filter=pads.field('episode_index') == ep['episode_index'], columns=['frame_index', 'observation.state', 'action']).sort_by('frame_index').to_pylist()
        for row in rows:
            saved = dataset[position]
            np.testing.assert_array_equal(saved['actions'], np.asarray(row['action'], np.float32))
            np.testing.assert_array_equal(saved['observations']['state'], np.asarray(row['observation.state'][:19], np.float32))
            np.testing.assert_array_equal(saved['observations']['source_state'], np.asarray(row['observation.state'], np.float32))
            position += 1
        assert dataset[position - 1]['dones'] == 1 and dataset[position - 1]['rewards'] == 1
    print(f'PASS exact state/action/source-state: {position} frames', flush=True)
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=args.action_horizon, action_dim=32, max_token_len=200)
    data_factory = UR5eDataConfig(repo_id=args.asset_id, assets=oc.AssetsConfig(assets_dir=str(args.checkpoint / 'assets'), asset_id=args.asset_id))
    train_config = dataclasses.replace(oc.get_config('expo_pi05_droid_full_finetune_sft_cartesian_state'), model=model_config, data=data_factory)
    buffer = PiReplayBuffer(example_action=dataset[0]['actions'][None], capacity=len(dataset)+args.action_horizon, pi_train_config=train_config, replan_steps=8, task_description=mapping['prompt'])
    buffer.seed(42)
    buffer.insert_dataset(dataset)
    sample = buffer.sample_jax(2, success_only=True)
    assert sample is not None
    assert np.all(buffer.dataset_dict['is_success'][:len(dataset)])
    for leaf in jax.tree.leaves(sample):
        if hasattr(leaf, 'dtype') and np.issubdtype(leaf.dtype, np.number):
            assert np.isfinite(np.asarray(leaf)).all()
    print('PASS normalization, tokenizer, buffer insertion and success-only sampling', flush=True)
    report = {'frames': position, 'checkpoint': str(args.checkpoint.resolve()), 'action_horizon': args.action_horizon,
              'data_exact_match': True, 'buffer_sample_passed': True, 'forward_executed': False}
    # Parameter names/shapes prove compatibility without trusting a config filename.
    params = _restore_pi05_params(str(args.checkpoint / 'params'))
    flat = traverse_util.flatten_dict(params)
    report['checkpoint_lora_tensors'] = sum('lora' in '/'.join(k) for k in flat)
    shape_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    expected = traverse_util.flatten_dict(nnx.state(shape_model).to_pure_dict())
    assert set(flat) == set(expected), (set(flat)-set(expected), set(expected)-set(flat))
    assert all(flat[k].shape == v.shape for k,v in expected.items())
    report['full_checkpoint_tensors'] = len(flat)
    lora_config = dataclasses.replace(model_config, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    lora_model = nnx.eval_shape(lora_config.create, jax.random.key(0))
    expected_lora = traverse_util.flatten_dict(nnx.state(lora_model).to_pure_dict())
    missing = set(expected_lora) - set(flat)
    assert missing and all('lora' in '/'.join(k) for k in missing)
    assert not (set(flat) - set(expected_lora))
    assert all(flat[k].shape == expected_lora[k].shape for k in flat)
    train_cfg_lora = dataclasses.replace(train_config, model=lora_config, freeze_filter=lora_config.get_freeze_filter())
    trainable = traverse_util.flatten_dict(nnx.state(lora_model).filter(train_cfg_lora.trainable_filter).to_pure_dict())
    paths = ['/'.join(k) for k in trainable]
    report['lora_missing_only_adapter_tensors'] = len(missing)
    report['trainable_paths'] = paths
    report['vlm_lora_trainable'] = any('lora' in k and 'llm' in k and '_1' not in k for k in paths)
    report['action_expert_lora_trainable'] = any('lora' in k and 'llm' in k and '_1' in k for k in paths)
    report['vision_encoder_trainable'] = any('img' in k for k in paths)
    print('PASS full checkpoint tree and LoRA base-weight compatibility', flush=True)
    print({k:v for k,v in report.items() if k != 'trainable_paths'}, flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    if args.forward:
        model = model_config.load(jax.tree.map(jnp.asarray, params), remove_extra_params=False)
        model.eval()
        raw = dataset[0]['observations'].copy()
        # The real replay buffer already applies SFT-compatible preprocessing.
        packed = {key: np.asarray(value) for key, value in raw.items()}
        packed['actions'] = np.zeros((args.action_horizon, 7), np.float32)
        transformed = buffer._model_transform(buffer._transform(packed))
        observation = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], {k:v for k,v in transformed.items() if k in ['image', 'image_mask', 'state', 'tokenized_prompt', 'tokenized_prompt_mask']}))
        actions = jax.jit(model.sample_actions, static_argnames=('num_steps','num_samples'))(jax.random.key(42), observation, num_steps=10, num_samples=1)
        actions = np.asarray(actions.block_until_ready()).reshape(1, args.action_horizon, 32)
        dc = data_factory.create(train_config.assets_dirs, model_config)
        raw_actions = transforms.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)({'actions': actions[0]})['actions'][:, :7]
        assert np.isfinite(actions).all() and np.isfinite(raw_actions).all()
        report.update(forward_executed=True, normalized_shape=list(actions.shape), raw_action_shape=list(raw_actions.shape), raw_action_abs_max=float(np.abs(raw_actions).max()))
        print('PASS GPU forward, 10 denoising steps:', report['raw_action_shape'], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
