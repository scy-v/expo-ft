"""Compare observation packing with cached real inputs; no model or robot.

Run on an idle inference GPU. Measures packing plus transfer to a ready GPU
result, excluding shared input transforms, disk reads, and initial compilation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from configs.model.expo_ft_ur_config import get_config
from expo_ft.agents.vla.pi05 import Pi05Agent, _cpu_pack_observation
from expo_ft.utils.train_utils import build_pi05_config


@jax.jit
def _jitted_pack_observation(inputs):
    """Preserved pre-change implementation for comparison only."""
    packed = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], inputs)
    packed['image'] = {
        key: value.astype(jnp.float32) / 255.0 * 2.0 - 1.0
        if value.dtype == jnp.uint8 else value
        for key, value in packed['image'].items()
    }
    return packed


def stats(values):
    a = np.asarray(values)
    return {key: float(value) for key, value in dict(
        mean=a.mean(), median=np.median(a), p95=np.percentile(a, 95),
        p99=np.percentile(a, 99), min=a.min(), max=a.max()).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/online_rl/success/10/traj.hdf5')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error('repeats and warmup must be positive')
    device = jax.devices('gpu')[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    _, cfg, _, _ = build_pi05_config(get_config())
    agent = object.__new__(Pi05Agent)
    agent.data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    pre, model = agent._build_input_transform_pipeline()
    cached = []
    with h5py.File(args.dataset) as f:
        obs = f['saved_observation']
        indices = np.linspace(0, len(obs['state']) - 1, 32, dtype=int)
        for index in indices:
            raw = {key: obs[key][index] for key in (
                'state', 'exterior_image_1_left', 'wrist_image_left', 'prompt')}
            if isinstance(raw['prompt'], bytes):
                raw['prompt'] = raw['prompt'].decode()
            raw['actions'] = np.zeros(7)
            cached.append(model(pre(raw)))

    def old(inputs):
        return _jitted_pack_observation(inputs)

    def new(inputs):
        return jax.device_put(_cpu_pack_observation(inputs), sharding)

    # Check all cached frames before timing; allow only float32 rounding error.
    max_error = 0.0
    for inputs in cached:
        a, b = jax.device_get((old(inputs), new(inputs)))
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert x.shape == y.shape and x.dtype == y.dtype
            np.testing.assert_allclose(x, y, rtol=0, atol=2e-7)
            if np.issubdtype(x.dtype, np.floating):
                max_error = max(max_error, float(np.max(np.abs(x-y))))
    samples = {'gpu_pack': [], 'cpu_pack_upload': []}
    functions = {'gpu_pack': old, 'cpu_pack_upload': new}
    for i in range(args.warmup + args.repeats):
        inputs = cached[i % len(cached)]
        # Alternate order to limit warm-clock/order bias; no overlap between calls.
        names = list(functions) if i % 2 else list(reversed(functions))
        for name in names:
            cache_size = _jitted_pack_observation._cache_size()
            start = time.perf_counter_ns()
            output = functions[name](inputs)
            jax.block_until_ready(output)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            assert _jitted_pack_observation._cache_size() == cache_size
            assert all(x.devices() == {device} for x in jax.tree.leaves(output))
            if i >= args.warmup:
                samples[name].append(elapsed_ms)
            del output
    result = dict(
        scope='Cached real transformed observations, batch1, packing plus upload and GPU completion; no model, concurrent learner, robot, disk I/O or common transforms in timer.',
        dataset=str(Path(args.dataset).resolve()), frame_indices=indices.tolist(),
        repeats=args.repeats, warmup=args.warmup, unit='ms', device=str(device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        source=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        pi05_source_sha256=hashlib.sha256((ROOT/'expo_ft/agents/vla/pi05.py').read_bytes()).hexdigest(),
        jax=jax.__version__, numpy=np.__version__, max_absolute_error=max_error,
        inputs=jax.tree.map(lambda x: dict(shape=list(np.shape(x)), dtype=str(np.asarray(x).dtype)), cached[0]),
        input_bytes=sum(np.asarray(x).nbytes for x in jax.tree.leaves(cached[0])),
        cpu_packed_bytes=sum(x.nbytes for x in jax.tree.leaves(_cpu_pack_observation(cached[0]))),
        statistics={name: stats(v) for name, v in samples.items()}, samples=samples)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','inputs')}, indent=2))


if __name__ == '__main__':
    main()
