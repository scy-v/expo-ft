"""CPU checks for detailed input tracing; no model or robot initialization."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from client.ur import timing


class Composite:
    def __init__(self, *stages):
        self.transforms = stages

    def __call__(self, data):
        for stage in self.transforms:
            data = stage(data)
        return data


class InputTimingTests(unittest.TestCase):
    def test_nested_pipeline_keeps_order_and_result(self):
        pipeline = Composite(lambda x: x + 2, Composite(lambda x: x * 3))
        for enabled in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                writer = timing.TraceWriter(Path(directory) / 'trace.jsonl') if enabled else None
                with patch.object(timing, '_writer', writer):
                    self.assertEqual(timing.transform_pipeline(pipeline, 4, 'pipeline'), 18)
                if writer:
                    writer.close()
                    rows = [json.loads(s) for s in writer.path.read_text().splitlines()]
                    self.assertEqual(sum(r['event'] == 'end' for r in rows), 4)
                    self.assertTrue(all(r['thread_cpu_ms'] >= 0 for r in rows if r['event'] == 'end'))

    def test_real_input_wrapper_preserves_outputs_and_only_waits_when_enabled(self):
        source = ast.parse(Path('expo_ft/agents/vla/pi05.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Pi05Agent')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process_raw_inputs')
        for enabled in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                writer = timing.TraceWriter(Path(directory) / 'trace.jsonl') if enabled else None
                waits = Mock()
                packed = dict(image=np.ones((1,2,2,3)), image_mask=True, state=np.arange(3))
                pack = Mock(return_value=packed)
                upload = Mock(return_value=packed)
                env = dict(np=np, span=timing.span, traced=timing.traced, enabled=timing.enabled,
                           transform_pipeline=timing.transform_pipeline,
                           jax=SimpleNamespace(block_until_ready=waits, device_put=upload), _cpu_pack_observation=pack)
                exec(compile(ast.Module(body=[method], type_ignores=[]), '<input-wrapper>', 'exec'), env)
                agent = SimpleNamespace(input_transforms=Composite(lambda x: x), input_model_transforms=Composite(lambda x: x), infer_sharding=object())
                raw = dict(image=np.ones((2,2,3))*8, state=np.arange(3))
                with patch.object(timing, '_writer', writer):
                    result = env['process_raw_inputs'](agent, raw, 7, 224)
                self.assertIs(result['image'], packed['image'])
                self.assertEqual(pack.call_count, 1)
                upload.assert_called_once_with(packed, agent.infer_sharding)
                self.assertEqual(waits.call_count, 2 if enabled else 0)
                self.assertEqual(raw['image'].dtype, np.uint8)
                if writer:
                    writer.close()
                    rows = [json.loads(s) for s in writer.path.read_text().splitlines()]
                    spans = {r['span'] for r in rows}
                    self.assertTrue({'inference.input.numpy_asarray', 'inference.input.image_uint8',
                                     'inference.input.pack_cpu', 'inference.input.upload_dispatch',
                                     'inference.input.upload_ready'} <= spans)
                    self.assertEqual(sum(r['event']=='begin' for r in rows), sum(r['event']=='end' for r in rows))
