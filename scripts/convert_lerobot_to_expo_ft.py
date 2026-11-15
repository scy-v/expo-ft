"""Export local LeRobot v2/v3 success demonstrations for EXPO-FT's HDF5 loader.

This maps storage fields, not robot kinematics or action units. Use an explicit
JSON mapping agreed with the SFT preprocessing and online task configuration.
By default, sample up to 10 episodes without replacement before reading frames.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
from pathlib import Path
import random
import shutil
import tempfile

import av
import h5py
import numpy as np
from PIL import Image
import pyarrow.dataset as pads
import pyarrow.parquet as pq


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def vector(row, spec):
    value = np.asarray(row[spec['key']], dtype=np.float32)
    if value.ndim > 1:
        raise ValueError(f"Expected per-frame vector, not an action chunk: {spec['key']} {value.shape}")
    value = value.reshape(-1)
    if 'indices' in spec:
        indices = spec['indices']
        if not indices or min(indices) < 0 or max(indices) >= len(value):
            raise ValueError(f"Invalid indices for {spec['key']}: {indices}, size={len(value)}")
        value = value[indices]
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite values in {spec['key']}")
    return value


def image_array(value, root):
    if isinstance(value, dict):
        source = io.BytesIO(value['bytes']) if value.get('bytes') is not None else local_path(root, value['path'])
        with Image.open(source) as img:
            return np.asarray(img.convert('RGB'))
    if isinstance(value, str):
        with Image.open(local_path(root, value)) as img:
            return np.asarray(img.convert('RGB'))
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}")
    if arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("Floating images must be finite RGB in [0, 1]")
        arr = np.rint(arr * 255).astype(np.uint8)
    if arr.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype: {arr.dtype}")
    return arr


def video_frames(path, times, tolerance):
    """Decode a contiguous episode window and match frames by presentation time."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((times[0] - tolerance) / float(stream.time_base))), stream=stream)
        decoded = iter(container.decode(stream))
        previous = None
        current = next(decoded, None)
        for target in times:
            while current is not None and float(current.time) < target:
                previous = current
                current = next(decoded, None)
            choices = [f for f in (previous, current) if f is not None]
            if not choices:
                raise ValueError(f"No video frame at {target}: {path}")
            selected = min(choices, key=lambda f: abs(float(f.time) - target))
            if abs(float(selected.time) - target) > tolerance:
                raise ValueError(f"Video/frame timestamp mismatch: {path}, expected={target}, got={selected.time}")
            yield selected.to_ndarray(format='rgb24')


def episode_metadata(root, info):
    if info['codebase_version'].startswith('v3.'):
        files = sorted((root / 'meta/episodes').rglob('*.parquet'))
        records = [row for file in files for row in pq.read_table(file).to_pylist()]
    elif info['codebase_version'].startswith('v2.'):
        records = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported LeRobot version: {info['codebase_version']}")
    result = {int(row['episode_index']): row for row in records}
    if len(result) != len(records) or not records:
        raise ValueError('Missing or duplicate episode metadata')
    return result


def validate_mapping(mapping):
    for key in ('prompt', 'action_space', 'gripper_action_space', 'action_semantics', 'state_semantics', 'images', 'observations', 'arm_action', 'gripper_action'):
        if not mapping.get(key):
            raise ValueError(f"Mapping must explicitly specify {key}")
    for name in (*mapping['images'], *mapping['observations'], mapping['action_space'], mapping['gripper_action_space']):
        if '/' in name or name in ('.', '..'):
            raise ValueError(f"Invalid HDF5 field name: {name}")
    if not {'exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left'} <= mapping['images'].keys():
        raise ValueError('All three EXPO-FT image fields must be explicitly mapped (side-camera duplication is allowed)')
    if 'gripper_position' not in mapping['observations']:
        raise ValueError('Missing gripper_position observation mapping')
    if not {'cartesian_position', 'joint_position'} & mapping['observations'].keys():
        raise ValueError('Provide cartesian_position or joint_position')
    if set(mapping['images']) & set(mapping['observations']) or 'prompt' in mapping['observations']:
        raise ValueError('Duplicate/reserved observation field names')


@contextmanager
def new_output(output):
    """Publish only complete exports; never overwrite an existing dataset."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        yield staging
        # Recheck before rename; no existing dataset is ever intentionally replaced.
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise


def select_episodes(metadata, *, num_episodes=10, seed=0, episodes=None, all_episodes=False):
    available = sorted(metadata)
    if episodes is not None:
        if all_episodes:
            raise ValueError('Cannot combine explicit episodes with all_episodes')
        selected = list(episodes)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Select nonempty, unique episode IDs')
        missing = set(selected) - set(available)
        if missing:
            raise ValueError(f'Unknown episode IDs: {sorted(missing)}')
        mode, requested_count = 'explicit', len(selected)
    elif all_episodes:
        selected = available
        mode, requested_count = 'all', len(available)
    else:
        if num_episodes <= 0:
            raise ValueError('num_episodes must be positive; use --all-episodes for all')
        selected = sorted(random.Random(seed).sample(available, min(num_episodes, len(available))))
        mode, requested_count = 'random', num_episodes
    return selected, {
        'mode': mode,
        'seed': seed if mode == 'random' else None,
        'requested_count': requested_count,
        'available_count': len(available),
        'selected_episode_ids': selected,
    }


def convert(root: Path, output: Path, mapping: dict, *, episodes=None, num_episodes=10,
            seed=0, all_episodes=False, assume_success=False):
    if not assume_success:
        raise ValueError('Explicit --assume-success required: the upstream loader labels every imported episode successful')
    root = root.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('Output must be outside the source dataset')
    validate_mapping(mapping)
    info = json.loads((root / 'meta/info.json').read_text())
    fps = float(info['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid dataset FPS')
    metadata = episode_metadata(root, info)
    selected, selection = select_episodes(metadata, num_episodes=num_episodes, seed=seed,
                                         episodes=episodes, all_episodes=all_episodes)
    print(f'Selected {len(selected)}/{len(metadata)} episodes ({selection["mode"]}, '
          f'seed={selection["seed"]}): {selected}', flush=True)
    files = sorted((root / 'data').rglob('*.parquet'))
    source = pads.dataset([str(f) for f in files], format='parquet')
    image_keys = set(mapping['images'].values())
    specs = [*mapping['observations'].values(), mapping['arm_action'], mapping['gripper_action']]
    numeric_keys = {spec['key'] for spec in specs}
    video_keys = {key for key in image_keys if info['features'][key]['dtype'] == 'video'}
    columns = sorted(numeric_keys | (image_keys - video_keys) | {'frame_index', 'timestamp', 'episode_index'})
    report = {'source': str(root), 'source_info': info, 'mapping': mapping,
              'all_episodes_assumed_success': True, 'selection': selection, 'episodes': []}
    with new_output(output) as staging:
        for episode in selected:
            meta = metadata[episode]
            # Filter all parquet fragments, including episodes split across file boundaries.
            table = source.to_table(columns=columns, filter=pads.field('episode_index') == episode)
            rows = table.sort_by('frame_index').to_pylist()
            length = int(meta['length'])
            if not rows or len(rows) != length:
                raise ValueError(f'Episode {episode}: metadata length {length}, read {len(rows)}')
            if [int(r['frame_index']) for r in rows] != list(range(length)):
                raise ValueError(f'Episode {episode}: duplicated, missing or nonzero-based frame indices')
            times = np.asarray([float(r['timestamp']) for r in rows])
            if not np.isfinite(times).all() or not np.allclose(times, np.arange(length) / fps, atol=1e-4):
                raise ValueError(f'Episode {episode}: timestamps do not match declared FPS')
            iterators = {}
            try:
                for key in video_keys:
                    if info['codebase_version'].startswith('v3.'):
                        chunk = int(meta[f'videos/{key}/chunk_index'])
                        file = int(meta[f'videos/{key}/file_index'])
                        offset = float(meta[f'videos/{key}/from_timestamp'])
                    else:
                        chunk = episode // int(info.get('chunks_size', 1000))
                        file, offset = 0, 0.0
                    relative = info['video_path'].format(video_key=key, episode_index=episode, episode_chunk=chunk, chunk_index=chunk, file_index=file)
                    iterators[key] = video_frames(local_path(root, relative), times + offset, tolerance=0.5 / fps + 1e-4)
                ep_dir = staging / str(episode)
                ep_dir.mkdir()
                with h5py.File(ep_dir / 'traj.hdf5', 'w') as h5:
                    obs = h5.create_group('saved_observation')
                    actions = h5.create_group('action')
                    h5.attrs['source_episode_index'] = episode
                    h5.attrs['fps'] = fps
                    obs.create_dataset('prompt', data=np.array([mapping['prompt']] * length, dtype=object), dtype=h5py.string_dtype())
                    for field, spec in mapping['observations'].items():
                        values = np.stack([vector(row, spec) for row in rows])
                        expected = {'cartesian_position': 6, 'joint_position': 7, 'gripper_position': 1}.get(field)
                        if expected is not None and values.shape[1] != expected:
                            raise ValueError(f'{field}: expected {expected} dimensions, got {values.shape[1]}')
                        obs.create_dataset(field, data=values[:, 0] if field == 'gripper_position' else values)
                    arm = np.stack([vector(row, mapping['arm_action']) for row in rows])
                    grip = np.stack([vector(row, mapping['gripper_action']) for row in rows])
                    if arm.shape[1] not in (6, 7) or grip.shape[1] != 1:
                        raise ValueError(f'Expected 6/7 arm + 1 gripper action, got {arm.shape}, {grip.shape}')
                    actions.create_dataset(mapping['action_space'], data=arm)
                    actions.create_dataset('gripper_' + mapping['gripper_action_space'], data=grip[:, 0])
                    for t, row in enumerate(rows):
                        decoded = {key: next(it) for key, it in iterators.items()}
                        for field, key in mapping['images'].items():
                            image = decoded[key] if key in decoded else image_array(row[key], root)
                            if 'image_size' in mapping:
                                height, width = mapping['image_size']
                                image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC))
                            if t == 0:
                                obs.create_dataset(field, shape=(length, *image.shape), dtype='uint8', chunks=(1, *image.shape), compression='lzf')
                            if image.shape != obs[field].shape[1:]:
                                raise ValueError(f'Episode {episode}: image shape changes at frame {t}')
                            obs[field][t] = image
                report['episodes'].append({'episode_index': episode, 'frames': length})
                print(f'Converted episode {episode}: {length} frames', flush=True)
            finally:
                for iterator in iterators.values():
                    iterator.close()
        (staging / 'conversion.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Direct parent of numeric episode directories, e.g. data/my_task/success')
    parser.add_argument('--mapping', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--num-episodes', type=int, default=10,
                           help='Randomly sample up to N episodes without replacement (default: 10)')
    selection.add_argument('--episodes', type=int, nargs='+', help='Convert only these original episode IDs')
    selection.add_argument('--all-episodes', action='store_true', help='Explicitly convert every episode')
    parser.add_argument('--seed', type=int, default=0, help='Random sampling seed (default: 0)')
    parser.add_argument('--assume-success', action='store_true')
    args = parser.parse_args()
    convert(args.dataset_root, args.output_dir, json.loads(args.mapping.read_text()),
            episodes=args.episodes, num_episodes=args.num_episodes, seed=args.seed,
            all_episodes=args.all_episodes, assume_success=args.assume_success)


if __name__ == '__main__':
    main()
