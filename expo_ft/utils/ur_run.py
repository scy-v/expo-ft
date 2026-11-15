"""Persist the resolved UR online model contract for full checkpoint evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess


def write_run_config(output_dir, config, task, replan_steps, *, resuming):
    if task.env_type != 'ur':
        return
    from expo_ft.utils.train_utils import build_pi05_config
    _, train, _, _ = build_pi05_config(config)
    stats = Path(train.data.assets.assets_dir) / train.data.assets.asset_id / 'norm_stats.json'
    payload = {'agent_config': config.to_dict(), 'replan_steps': replan_steps,
               'task_description': task.language_instruction, 'control_hz': task.control_hz,
               'norm_stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
               'critic_camera_keys': list(getattr(task, 'critic_camera_keys', ('base_0_rgb', 'left_wrist_0_rgb'))),
               'source_files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for base in ('client/ur', 'client/envs', 'configs', 'expo_ft') for p in Path(base).rglob('*.py') if 'openpi/' not in str(p)},
               'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               'working_tree_modified': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())}
    # JSON represents tuple-valued model fields as lists; compare the serialized contract.
    payload = json.loads(json.dumps(payload))
    path = Path(output_dir) / 'ur_run.json'
    if resuming and path.exists():
        previous = json.loads(path.read_text())
        for key in ['agent_config','replan_steps','control_hz','norm_stats_sha256','critic_camera_keys']:
            if previous[key] != payload[key]:
                raise ValueError(f'UR resume configuration changed: {key}')
    else:
        path.write_text(json.dumps(payload,indent=2)+'\n')
