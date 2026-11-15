"""OpenPI input transforms for UR5e HDF5 exports with the original SFT state.

The HDF5 envelope matches the DROID dataset reader, but UR actions remain
Cartesian deltas plus gripper position and must never go to a DROID controller.
"""
import dataclasses

import numpy as np
from openpi import transforms
from openpi.training import config as openpi_config


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (19,):
            raise ValueError(f'UR5e SFT state must have 19 dimensions, got {state.shape}')
        base, wrist = data['exterior_image_1_left'], data['wrist_image_left']
        result = {
            'state': state,
            'image': {'base_0_rgb': base, 'left_wrist_0_rgb': wrist, 'right_wrist_0_rgb': np.zeros_like(base)},
            'image_mask': {'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_, 'right_wrist_0_rgb': np.False_},
            'prompt': data['prompt'],
        }
        if 'actions' in data:
            result['actions'] = data['actions']
        return result


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {'actions': np.asarray(data['actions'])[..., :7]}


@dataclasses.dataclass(frozen=True)
class UR5eDataConfig(openpi_config.DataConfigFactory):
    """No extra delta conversion: exported actions already contain deltas."""
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(),
            data_transforms=transforms.Group(inputs=[UR5eInputs()], outputs=[UR5eOutputs()]),
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )


def make_train_config(*, lora=False, action_horizon=50):
    """UR state/action contract, with explicit full or dual-LoRA architecture."""
    from openpi.models.pi0_config import Pi0Config
    model = Pi0Config(
        pi05=True, action_dim=32, action_horizon=action_horizon, max_token_len=200,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    base = openpi_config.get_config("expo_pi05_droid_full_finetune_sft_cartesian_state")
    return dataclasses.replace(
        base, name="expo_pi05_ur_lora" if lora else "expo_pi05_ur_full",
        model=model, data=UR5eDataConfig(), freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
    )
