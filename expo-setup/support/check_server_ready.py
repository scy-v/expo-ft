"""Validate the documented server command without starting a learner or robot.

Run from the repository root after sourcing expo-setup/activate.sh.
Uses CPU only; writes an offline W&B diagnostic and a local JSON report.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['WANDB_MODE'] = 'offline'
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import h5py
import numpy as np
import yaml
from etils import epath
report = {'scope': 'CPU server configuration, dependencies and local services; no learner/model forward/robot'}
text = Path('expo-setup/训练命令.md').read_text()
command = next(b for b in re.findall(r'```bash\n(.*?)```', text, re.S) if 'python train_pi_robo.py' in b)
parts = shlex.split(command.replace('\\\n', ' '))
argv = parts[parts.index('train_pi_robo.py'):]
sys.argv = argv
import train_pi_robo as training
from expo_ft.utils.train_utils import build_pi05_config, init_wandb
from expo_ft.utils.ur_run import write_run_config
import wandb

training.FLAGS(argv)
# Capture the launcher arguments using a shell function, never executing Python/learner.
captured = subprocess.check_output(['bash', '-c', r'''unset SFT_CKPT DATASET_PATH ASSET_ID RUN_NAME
python() { printf '%s\0' "$@"; }
export -f python
bash scripts/ur/run_server.sh'''])
launcher = captured.decode().strip('\0').split('\0')
assert launcher[0] == argv[0] and sorted(launcher[1:]) == sorted(argv[1:])
report['launcher_matches_document'] = True
f = training.FLAGS
cfg, task = f.config, f.config_task
_, train, _, model_cls = build_pi05_config(cfg)
assert model_cls == 'EXPOLearner' and cfg.pi05_config_name == 'expo_pi05_ur_lora'
from configs.model.expo_ft_pi_config import get_config as author_agent_config
from openpi.training import config as openpi_config
reference = author_agent_config()
for key, value in reference.items():
    if not key.startswith('pi05_'):
        assert cfg[key] == value, (key, cfg[key], value)
author_train = openpi_config.get_config('expo_pi05_droid_lora_finetune_sft_cartesian_state')
assert train.lr_schedule == author_train.lr_schedule
assert train.optimizer == author_train.optimizer
assert train.ema_decay == author_train.ema_decay
assert train.model.paligemma_variant == 'gemma_2b_lora'
assert train.model.action_expert_variant == 'gemma_300m_lora'
assert train.model.action_dim == 32 and train.model.max_token_len == 200
assert cfg.valids_keep_terminal_windows is False
assert task.edit_action_xyzg is False
report['author_algorithm_config_matches'] = True
report['pi05_learning_rate'] = float(train.lr_schedule.create()(0))

assert task.env_type == 'ur' and task.control_hz == 10
assert train.model.action_horizon == 50 and f.replan_steps == 8 and f.delay == 0
assert f.batch_size == 64 and f.utd_ratio == 20 and f.num_updates == 3
assert f.max_steps == 100000
assert f.update_type == 'episode' and f.offline_ratio == 0 and f.fsdp_devices == 1
assert Path(cfg.pi05_weight_loader_path).resolve() == (ROOT/'checkpoints/sft/ur_insert/2500/params').resolve()
assert train.weight_loader.params_path == cfg.pi05_weight_loader_path
stats = Path(cfg.pi05_assets_dir)/cfg.pi05_asset_id/'norm_stats.json'
norm = json.loads(stats.read_text())['norm_stats']
assert len(norm['state']['mean']) == 19 and len(norm['actions']['mean']) == 7
for group in norm.values():
    for values in group.values():
        assert np.isfinite(np.asarray(values)).all()
conversion = json.loads((Path(f.dataset_path)/'conversion.json').read_text())
assert len(conversion['episodes']) == f.num_data == 10
for episode in conversion['episodes']:
    with h5py.File(Path(f.dataset_path)/str(episode['episode_index'])/'traj.hdf5') as data:
        assert data['saved_observation/state'].shape == (episode['frames'], 19)
        assert data['action/cartesian_delta'].shape == (episode['frames'], 6)
        assert data['saved_observation/prompt'].asstr()[0] == task.language_instruction
rollout = yaml.safe_load(Path(task.rollout_config_path).read_text())
assert type(rollout['warmup']) is bool and type(rollout['sft_only']) is bool
if rollout['sft_only']:
    assert f.delay == 0
report['rollout'] = rollout
hardware = yaml.safe_load(Path(task.ur_config_path).read_text())
assert hardware['robot']['action_fps'] == task.control_hz
assert hardware['robot']['force_mode']['rtde_freq'] == 100
assert hardware['online']['max_episode_chunks'] == 60
assert 'model' not in hardware and 'policy' not in hardware
assert not {'num_episodes', 'episode_timeout_sec'} & hardware.get('record', {}).keys()
assert hardware['task']['description'] == task.language_instruction
assert 'auto_reset_steps' not in task
report['episode_budget'] = dict(max_episode_chunks=60, replan_steps=f.replan_steps,
                               executed_action_limit=60*f.replan_steps, wall_clock_timeout=False)
report['yaml_scope'] = 'repository copy; not a live attestation of remote actor configuration'
run = Path(f.output_dir)/f.run_name
assert not run.exists(), f'Fresh-run output already exists: {run}; choose new name or explicit resume'
for name in ['insert_sft.yaml', 'insert_expo.yaml']:
    evaluation = yaml.safe_load((Path('configs/ur')/name).read_text())['model']
    assert evaluation['action_horizon'] == train.model.action_horizon
    assert evaluation['replan_steps'] == f.replan_steps
    assert evaluation['kind'] == ('sft' if name == 'insert_sft.yaml' else 'expo')
    assert evaluation['architecture'] == ('full' if name == 'insert_sft.yaml' else 'lora')
    assert Path(evaluation['norm_stats_path']).resolve() == stats.resolve()
    if name == 'insert_expo.yaml':
        assert Path(evaluation['run_config']).resolve() == (run/'ur_run.json').resolve()
        assert Path(evaluation['checkpoint_dir']).resolve() == (run/'checkpoints').resolve()
for folder in ['data', 'checkpoints', 'logs', 'expo-setup/uv-cache', 'expo-setup/outputs']:
    path = Path(folder).resolve()
    assert path.is_relative_to('/data/suchenyu/expo-ft'), (folder,path)
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b'write probe')
with socket.socket() as sock:
    sock.bind((f.client_host, f.client_port))
report['port_bind'] = f'{f.client_host}:{f.client_port}'
with tempfile.TemporaryDirectory(prefix='.server-check-', dir=ROOT/'checkpoints') as tmp:
    from expo_ft.utils.train_utils import set_compilation_cache_dir
    cache = Path(set_compilation_cache_dir("server-preflight")).resolve()
    assert cache.is_relative_to("/data/suchenyu/expo-ft")
    write_run_config(tmp, cfg, task, f.replan_steps, resuming=False)
    write_run_config(tmp, cfg, task, f.replan_steps, resuming=True)
    init_wandb(epath.Path(tmp), False, 'expo_ft_server_checks', 'server-preflight-20261116')
    try:
        wandb.config.update(f.flag_values_dict(), allow_val_change=False)
        report['wandb_mode'] = wandb.run.settings.mode
        assert report['wandb_mode'] == 'offline'
    finally:
        wandb.finish()
for package in ['jax','jaxlib','flax','torch','numpy','openpi','openpi-client','websockets']:
    report.setdefault('versions', {})[package] = importlib.metadata.version(package)
for module in ['openpi', 'openpi_client', 'lerobot', 'flax', 'optax', 'h5py', 'cv2',
               'tensorflow_probability.substrates.jax', 'expo_ft.agents.alg',
               'expo_ft.data.replay_buffer', 'expo_ft.env.env_client', 'expo_ft.agents.vla.pi05']:
    importlib.import_module(module)
report.update(command=command.strip(), config_passed=True, imports_passed=True,
              storage_write_passed=True, run_manifest_passed=True,
              frames=sum(e['frames'] for e in conversion['episodes']),
              checkpoint=str(Path(cfg.pi05_weight_loader_path).resolve()),
              output=str(run.resolve()), python=sys.version,
              gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True),
              repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              openpi_commit=subprocess.check_output(['git','-C','expo_ft/agents/vla/openpi','rev-parse','HEAD'],text=True).strip(),
              gpu_model_forward_tested=False, gradient_update_tested=False, robot_tested=False)
out=Path('expo-setup/support/logs/server-ready-20261116.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print('PASS documented arguments, 2500-step paths, normalization, 10 demos, task/model contract, output paths, imports, storage writes, port bind, offline W&B and manifest roundtrip',flush=True)
print('GPU model forward / gradient / robot NOT tested; inspect GPU snapshot and remaining validation.',flush=True)
print(out,flush=True)
