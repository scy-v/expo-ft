"""File-format regression tests with actual Parquet, video and HDF5 I/O."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import h5py
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('converter', ROOT / 'scripts/convert_lerobot_to_expo_ft.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
MAPPING = json.loads((ROOT / 'expo-setup/support/lerobot-ur5e-insert.json').read_text())


def png(color):
    out = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), color, np.uint8)).save(out, format='PNG')
    return {'bytes': out.getvalue(), 'path': None}


class ConversionTest(unittest.TestCase):
    def multi_episode_fixture(self, root, count=20):
        rows = self.fixture(root)
        ids = [11 + 7 * i for i in range(count)]
        records = [dict(row, episode_index=episode) for episode in ids for row in rows]
        for file in (root / 'data').glob('*.parquet'):
            file.unlink()
        pq.write_table(pa.Table.from_pylist(records), root / 'data/episodes.parquet')
        (root / 'meta/episodes.jsonl').write_text(''.join(
            json.dumps({'episode_index': episode, 'length': len(rows)}) + '\n' for episode in ids))
        return ids

    def fixture(self, root, *, version='v2.1', broken=False):
        (root / 'meta').mkdir(parents=True)
        (root / 'data').mkdir()
        features = {key: {'dtype': 'image' if version.startswith('v2') else 'video'} for key in set(MAPPING['images'].values())}
        info = dict(codebase_version=version, fps=10, features=features, video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
        (root / 'meta/info.json').write_text(json.dumps(info))
        rows = []
        for t in range(4):
            row = dict(episode_index=2, frame_index=t, timestamp=t / 10,
                       **{'observation.state': list(np.arange(48, dtype=float) + t), 'action': list(np.arange(7, dtype=float) + t)})
            if version.startswith('v2'):
                row.update({key: png(40 + t * 10) for key in features})
            rows.append(row)
        if broken:
            rows[2]['frame_index'] = 1
        # Intentionally split and reverse fragments to test chronological assembly.
        pq.write_table(pa.Table.from_pylist(rows[2:]), root / 'data/a.parquet')
        pq.write_table(pa.Table.from_pylist(rows[:2]), root / 'data/b.parquet')
        meta = {'episode_index': 2, 'length': 4}
        if version.startswith('v2'):
            (root / 'meta/episodes.jsonl').write_text(json.dumps(meta) + '\n')
        else:
            for key in features:
                meta.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0, f'videos/{key}/from_timestamp': .2})
                file = root / f'videos/{key}/chunk-000/file-000.mp4'
                file.parent.mkdir(parents=True)
                with av.open(str(file), 'w') as video:
                    stream = video.add_stream('libx264', rate=10)
                    stream.width = stream.height = 16
                    stream.pix_fmt = 'yuv420p'
                    for t in range(8):
                        frame = av.VideoFrame.from_ndarray(np.full((16,16,3), 40 + t * 10, np.uint8), format='rgb24')
                        for packet in stream.encode(frame):
                            video.mux(packet)
                    for packet in stream.encode():
                        video.mux(packet)
            (root / 'meta/episodes').mkdir()
            pq.write_table(pa.Table.from_pylist([meta]), root / 'meta/episodes/file.parquet')
        return rows

    def test_v2_roundtrip_loader(self):
        from types import SimpleNamespace
        from expo_ft.env.droid_utils import process_droid_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'src', Path(tmp) / 'out'
            rows = self.fixture(root)
            converter.convert(root, out, MAPPING, assume_success=True)
            data = process_droid_dataset(str(out), SimpleNamespace(action_space='cartesian_delta', gripper_action_space='position'))
            self.assertEqual(len(data), 4)
            np.testing.assert_array_equal([r['actions'] for r in data], [r['action'] for r in rows])
            np.testing.assert_array_equal(data[2]['observations']['state'], rows[2]['observation.state'][:19])
            self.assertEqual([r['rewards'] for r in data], [0,0,0,1])
            self.assertEqual([r['masks'] for r in data], [1,1,1,0])
            self.assertEqual(data[0]['observations']['prompt'], MAPPING['prompt'])
            with self.assertRaises(FileExistsError):
                converter.convert(root, out, MAPPING, assume_success=True)

    def test_v3_shared_video_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'src', Path(tmp) / 'out'
            self.fixture(root, version='v3.0')
            converter.convert(root, out, MAPPING, assume_success=True)
            with h5py.File(out / '2/traj.hdf5') as f:
                images = f['saved_observation/exterior_image_1_left'][:]
                np.testing.assert_allclose(images.mean(axis=(1,2,3)), [60,70,80,90], atol=3)
                np.testing.assert_array_equal(images, f['saved_observation/exterior_image_2_left'][:])

    def test_invalid_frames_leave_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'src', Path(tmp) / 'out'
            self.fixture(root, broken=True)
            with self.assertRaisesRegex(ValueError, 'frame indices'):
                converter.convert(root, out, MAPPING, assume_success=True)
            self.assertFalse(out.exists())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['src'])

    def test_success_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, 'assume-success'):
            converter.convert(Path('missing'), Path('out'), MAPPING)

    def test_cli_defaults_to_ten_random_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'src', Path(tmp) / 'out'
            ids = self.multi_episode_fixture(root)
            subprocess.run([
                sys.executable, str(ROOT / 'scripts/convert_lerobot_to_expo_ft.py'),
                '--dataset-root', str(root), '--output-dir', str(out),
                '--mapping', str(ROOT / 'expo-setup/support/lerobot-ur5e-insert.json'),
                '--assume-success',
            ], check=True, capture_output=True, text=True)
            report = json.loads((out / 'conversion.json').read_text())
            chosen = report['selection']['selected_episode_ids']
            self.assertEqual(len(chosen), 10)
            self.assertEqual(len(set(chosen)), 10)
            self.assertTrue(set(chosen) <= set(ids))
            self.assertNotEqual(chosen, ids[:10])
            self.assertEqual(report['selection']['seed'], 0)
            self.assertEqual({int(p.parent.name) for p in out.glob('*/traj.hdf5')}, set(chosen))
            self.assertEqual([e['episode_index'] for e in report['episodes']], chosen)

    def test_random_subset_reproducible_and_skips_unselected_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'src'
            ids = self.multi_episode_fixture(root)
            first = converter.convert(root, Path(tmp) / 'first', MAPPING,
                                      num_episodes=3, seed=17, assume_success=True)
            chosen = first['selection']['selected_episode_ids']
            # Unselected images cannot be decoded: conversion must never touch them.
            file = root / 'data/episodes.parquet'
            rows = pq.read_table(file).to_pylist()
            for row in rows:
                if row['episode_index'] not in chosen:
                    for key in set(MAPPING['images'].values()):
                        row[key] = {'bytes': None, 'path': 'does-not-exist.png'}
            pq.write_table(pa.Table.from_pylist(rows), file)
            second = converter.convert(root, Path(tmp) / 'second', MAPPING,
                                       num_episodes=3, seed=17, assume_success=True)
            self.assertEqual(first, second)
            self.assertEqual(len(chosen), 3)
            self.assertEqual(len(list((Path(tmp) / 'second').glob('*/traj.hdf5'))), 3)
            different, _ = converter.select_episodes(dict.fromkeys(ids), num_episodes=3, seed=18)
            self.assertNotEqual(chosen, different)

    def test_small_dataset_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'src'
            ids = self.multi_episode_fixture(root, count=3)
            sampled = converter.convert(root, Path(tmp) / 'sampled', MAPPING, assume_success=True)
            self.assertEqual(sampled['selection']['selected_episode_ids'], ids)
            self.assertEqual(sampled['selection']['requested_count'], 10)
            explicit = converter.convert(root, Path(tmp) / 'explicit', MAPPING,
                                         episodes=[ids[-1]], assume_success=True)
            self.assertEqual(explicit['selection']['selected_episode_ids'], [ids[-1]])
            full = converter.convert(root, Path(tmp) / 'full', MAPPING,
                                     all_episodes=True, assume_success=True)
            self.assertEqual(full['selection']['selected_episode_ids'], ids)
            self.assertEqual(full['selection']['mode'], 'all')

    def test_invalid_selection_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'src', Path(tmp) / 'out'
            self.fixture(root)
            for kwargs in ({'num_episodes': 0}, {'num_episodes': -1},
                           {'episodes': []}, {'episodes': [2, 2]}, {'episodes': [999]},
                           {'episodes': [2], 'all_episodes': True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    converter.convert(root, out, MAPPING, assume_success=True, **kwargs)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
