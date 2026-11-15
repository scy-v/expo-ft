"""UR insert task: standard EXPO rollout at 10 Hz, no separate NUC."""
import numpy as np
from ml_collections import ConfigDict


def get_config():
    from client.envs.ur_env import UREnv
    return ConfigDict(dict(
        env=UREnv, env_name='ur_insert', env_type='ur',
        ur_config_path='configs/ur/insert.yaml',
        language_instruction='Insert the black rack into the blue slot',
        action_space='cartesian_delta', gripper_action_space='position',
        example_action=np.array([[0., 0., 0., 0., 0., 0., 1.]]),
        control_hz=10, auto_reset_steps=600, edit_action_xyzg=False,
    ))
