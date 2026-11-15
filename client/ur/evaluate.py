"""UR evaluation on the robot host, or model-only checks/serving on the GPU host."""
import argparse
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ur/insert.yaml'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Validate config and weight paths; never connect hardware')
    mode.add_argument('--check-model', action='store_true', help='Load actual weights and infer on synthetic observations; no hardware')
    mode.add_argument('--serve', action='store_true', help='Serve the selected policy on the GPU host; no hardware')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['policy'].get('cuda_visible_devices', '0'))
    from .policy import URPolicy, check_model, validate_config
    if args.check:
        validate_config(config)
        print('PASS config, checkpoint paths and normalization; no hardware connected')
    elif args.check_model:
        check_model(config)
    elif args.serve:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        policy = URPolicy(config)
        # Compile before accepting a robot connection.
        from .policy import example_observation
        policy.infer(example_observation())
        WebsocketPolicyServer(policy, host=config['policy'].get('bind_host','0.0.0.0'),
            port=config['policy']['port'], metadata={'kind': config['model']['kind'],
            'architecture': config['model']['architecture'], 'replan_steps': config['model']['replan_steps']}).serve_forever()
    else:
        validate_config(config, files=config['policy'].get('location') != 'remote')
        from .ur import Inference
        Inference(args.config).run()


if __name__ == '__main__':
    main()
