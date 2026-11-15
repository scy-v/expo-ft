"""Standard EXPO-FT with UR state19, initialized from full SFT weights."""
from configs.model.expo_ft_pi_config import get_config as base_config


def get_config():
    config = base_config()
    config.pi05_config_name = "expo_pi05_ur_lora"
    config.pi05_action_horizon = 50
    config.pi05_weight_loader_path = "./checkpoints/sft/ur_insert/2500/params"
    config.pi05_assets_dir = "./checkpoints/sft/ur_insert/2500/assets"
    config.pi05_asset_id = "scylearning/insert_black_rack_into_blue_slot_20260916_v01"
    return config
