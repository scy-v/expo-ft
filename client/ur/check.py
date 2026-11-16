"""Actor preflight: imports/config by default; optional device inventory and TCP probes.

Never constructs a robot controller, opens the gripper, or reads keyboard events.
Only --camera-frames starts camera capture. Run from the repository root.
"""
import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml


def record(checks, name, action):
    try:
        detail = action()
    except Exception as exc:
        checks.append(dict(name=name, status='fail', detail=f'{type(exc).__name__}: {exc}'))
    else:
        checks.append(dict(name=name, status='pass', detail=detail))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(path):
    from .policy import validate_config
    from .keyboard_focus import validate_terminal_keyboard
    config = yaml.safe_load(Path(path).read_text())
    if 'online' in config:
        require('model' not in config and 'policy' not in config,
                'Online actor config must not contain evaluation model/policy settings')
        chunks = config['online']['max_episode_chunks']
        require(type(chunks) is int and chunks > 0, 'online.max_episode_chunks must be a positive integer')
        require(config['robot']['action_fps'] == 10, 'Expected 10 Hz actions')
        force = config['robot']['force_mode']
        require(force['rtde_freq'] == 100 and force['continuous_tracking'],
                'Expected continuous 100 Hz UR force tracking')
        require(not {'num_episodes', 'episode_timeout_sec'} & config.get('record', {}).keys(),
                'Episode count and timeout belong in evaluation config')
    else:
        validate_config(config, files=False)
        require(config['policy']['location'] == 'remote', 'Actor preflight requires policy.location=remote')
    validate_terminal_keyboard(config.get('keyboard_terminal_only', False))
    cameras = config['cameras']
    require(cameras['wrist_cam_serial'] != cameras['exterior_cam_serial'], 'Camera serials must be distinct')
    return config


def check_keyboard():
    from evdev import InputDevice, ecodes
    override = os.environ.get('EXPO_KEYBOARD_DEVICE')
    paths = [Path(override)] if override else sorted(Path('/dev/input').glob('event*'))
    candidates, denied = [], []
    for path in paths:
        try:
            device = InputDevice(str(path))
        except OSError as exc:
            denied.append(f'{path}: {exc}')
            continue
        try:
            keys = device.capabilities().get(ecodes.EV_KEY, [])
            if all(getattr(ecodes, key) in keys for key in ('KEY_A', 'KEY_B', 'KEY_C', 'KEY_Q')):
                candidates.append(dict(path=str(path), name=device.name))
        finally:
            device.close()
    require(len(candidates) == 1,
            f'Expected one readable keyboard; candidates={candidates}; inaccessible={denied}. '
            'Set EXPO_KEYBOARD_DEVICE to the intended /dev/input/by-id/...-event-kbd.')
    return candidates


def check_serial(path):
    require(Path(path).exists(), f'Missing gripper device: {path}')
    require(os.access(path, os.R_OK | os.W_OK), f'No read/write permission: {path}')
    return dict(path=path, target=str(Path(path).resolve()), opened=False)


def camera_inventory():
    import pyrealsense2 as rs
    return [dict(serial=d.get_info(rs.camera_info.serial_number),
                 name=d.get_info(rs.camera_info.name),
                 usb=d.get_info(rs.camera_info.usb_type_descriptor))
            for d in rs.context().query_devices()]


def check_cameras(config):
    devices = camera_inventory()
    actual = {d['serial'] for d in devices}
    expected = {str(config['wrist_cam_serial']), str(config['exterior_cam_serial'])}
    require(expected <= actual, f'Missing cameras {sorted(expected - actual)}; detected={devices}')
    return devices


def camera_frame(config, serial):
    import numpy as np
    from .cameras import RealSenseCamera
    camera = RealSenseCamera(serial, config['fps'], config['width'], config['height'])
    try:
        camera.connect()
        frame = camera.async_read()
        require(frame.shape == (config['height'], config['width'], 3), f'Unexpected shape {frame.shape}')
        require(frame.dtype == np.uint8, f'Unexpected dtype {frame.dtype}')
        return dict(serial=str(serial), shape=list(frame.shape), dtype=str(frame.dtype))
    finally:
        camera.disconnect()


def tcp_probe(host, port):
    with socket.create_connection((host, port), timeout=3):
        return dict(host=host, port=port, scope='TCP only; no policy or environment RPC')


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ur/insert.yaml'))
    parser.add_argument('--task-config', default='configs/task/ur_insert.py')
    parser.add_argument('--devices', action='store_true', help='Enumerate cameras/keyboard and check serial permissions')
    parser.add_argument('--camera-frames', action='store_true', help='Capture one RGB frame per configured camera; requires free cameras')
    parser.add_argument('--server-host', help='Optional TCP reachability probe; does not create an environment')
    parser.add_argument('--server-port', type=int, default=8102)
    parser.add_argument('--output', type=Path, help='Write the same JSON report to this path')
    args = parser.parse_args(argv)
    checks = []
    report = dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  platform=platform.platform(), python=sys.version, executable=sys.executable,
                  environment=str(Path(sys.prefix).resolve()), checks=checks)
    source_paths = [*Path('client').rglob('*.py'), *Path('tests').glob('test_ur*.py'),
                    *Path('configs/ur').glob('*.yaml'), Path(args.task_config),
                    Path('client/requirements-ur.lock')]
    report['source_sha256'] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(source_paths)) if path.is_file() and '.venv' not in path.parts
    }
    for label, path in [('repo_commit', '.'), ('openpi_commit', 'expo_ft/agents/vla/openpi')]:
        record(checks, label, lambda path=path: revision(path))
    for module in ['numpy', 'scipy', 'yaml', 'ml_collections', 'tyro', 'websockets',
                   'rtde_control', 'rtde_receive', 'pyDHgripper', 'serial', 'crcmod',
                   'pyrealsense2', 'evdev', 'Xlib', 'cv2', 'av', 'PIL', 'matplotlib',
                   'openpi_client.msgpack_numpy', 'client.run_client', 'client.ur.ur',
                   'client.ur.online_backend']:
        record(checks, f'import:{module}', lambda module=module: str(importlib.import_module(module).__file__))
    report['versions'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    config = None
    try:
        config = check_config(args.config)
    except Exception as exc:
        checks.append(dict(name='config', status='fail', detail=str(exc)))
    else:
        checks.append(dict(name='config', status='pass', detail=dict(path=str(args.config),
                      robot=config['robot'], cameras=config['cameras'], online=config.get('online'), model=config.get('model'))))
    def task_check():
        from client.run_client import load_task_config
        task = load_task_config(args.task_config)
        require(task.env_type == 'ur' and task.control_hz == 10, 'Expected standard UR task at 10 Hz')
        check_config(task.ur_config_path)
        return dict(path=args.task_config, hardware_config=task.ur_config_path)
    record(checks, 'task_config', task_check)
    if args.devices:
        record(checks, 'keyboard', check_keyboard)
        if config:
            record(checks, 'gripper_permissions', lambda: check_serial(config['robot']['gripper_port']))
            record(checks, 'cameras', lambda: check_cameras(config['cameras']))
    if args.camera_frames and config:
        for key in ('wrist_cam_serial', 'exterior_cam_serial'):
            record(checks, f'frame:{key}', lambda key=key: camera_frame(config['cameras'], config['cameras'][key]))
    if args.server_host:
        record(checks, 'server_tcp', lambda: tcp_probe(args.server_host, args.server_port))
    report['ok'] = all(c['status'] == 'pass' for c in checks)
    report['scope'] = 'No robot/gripper commands, model inference, or keyboard event reading'
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
