"""Exercise the registered UR task/model configs on the converted real episode."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import tempfile
import json
import numpy as np
import yaml
from configs.model.expo_ft_ur_config import get_config
from configs.task.ur_insert import get_config as get_task
from expo_ft.agents.vla.ur5e_data import make_train_config
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.ur_run import write_run_config

cfg=yaml.safe_load(Path('configs/ur/insert.yaml').read_text())
stats=Path(cfg['model']['norm_stats_path'])
model=get_config()
model.pi05_assets_dir=str(stats.parent)
model.pi05_asset_id='.'
model.pi05_weight_loader_path=str(Path(cfg['model']['checkpoint_dir'])/'params')
task=get_task()
data=process_droid_dataset('expo-setup/outputs/ur5e-insert-conversion-check',task,num_data=1)
for architecture in ['full','lora']:
    model.pi05_config_name='expo_pi05_ur_'+architecture
    rb=create_replay_buffer(config=model,example_action=data[0]['actions'][None],capacity=len(data)+50,
        task_description=task.language_instruction,replan_steps=8,seed=0)
    rb.insert_dataset(data)
    batch=rb.sample_jax(2,success_only=True)
    assert batch is not None
    assert np.all(rb.dataset_dict['is_success'][:len(data)])
    with tempfile.TemporaryDirectory() as directory:
        write_run_config(directory,model,task,8,resuming=False)
        saved=json.loads((Path(directory)/'ur_run.json').read_text())
        assert saved['agent_config']['pi05_config_name']==model.pi05_config_name
        assert saved['control_hz']==10
        write_run_config(directory,model,task,8,resuming=True)
    print(f'PASS registered {architecture} config: {len(data)} real frames, replay + manifest',flush=True)
    del rb
