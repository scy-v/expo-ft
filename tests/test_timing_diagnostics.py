import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from client.ur import timing


class TimingDiagnosticsTests(unittest.TestCase):
    def test_delayed_camera_is_identified_without_changing_observation(self):
        import time
        from client.ur.ur import Inference
        engine = Inference.__new__(Inference)
        engine.rtde_r = None
        engine.task_description = ''
        engine.fix_gripper = True
        marker = object()
        def read():
            time.sleep(0.02)
            return marker
        camera = Mock()
        camera.read.side_effect = read
        engine.cameras = {'wrist': camera}
        engine._transfer_obs_state = lambda obs: obs
        with tempfile.TemporaryDirectory() as directory:
            writer = timing.TraceWriter(Path(directory) / 'trace.jsonl')
            with patch.object(timing, '_writer', writer):
                self.assertIs(engine.get_obs_state()['wrist'], marker)
            writer.close()
            rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
            event = next(r for r in rows if r['span'] == 'camera.read' and r['event'] == 'end')
            self.assertEqual(event['camera'], 'wrist')
            self.assertGreaterEqual(event['duration_ms'], 15)
            camera.read.assert_called_once()

    def test_gpu_update_wait_precedes_weight_copy_only_when_enabled(self):
        # Exercise the real Python wrapper without importing a GPU model.
        import ast
        from contextlib import nullcontext
        source = ast.parse(Path('expo_ft/agents/alg/expo_ft.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'EXPOLearner')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        for enabled in [False, True]:
            events = []
            agent = Mock()
            agent.replace.return_value = agent
            agent._update_jit.side_effect = lambda *a: (events.append('dispatch') or agent, {})
            agent.cache_infer_params.side_effect = lambda: events.append('copy') or agent
            namespace = dict(span=lambda *a: nullcontext(), enabled=lambda: enabled,
                             jax=Mock(block_until_ready=lambda x: events.append('wait')))
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<update-wrapper>', 'exec'), namespace)
            namespace['update'](agent, agent, {}, 20, {})
            self.assertEqual(events, ['dispatch', 'wait', 'copy'] if enabled else ['dispatch', 'copy'])

    def test_disabled_does_not_create_files_or_change_result(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(timing, '_writer', None):
            timing.configure(False, directory, 'test')
            fn = timing.traced('no-op')(lambda x: x + 1)
            self.assertEqual(fn(2), 3)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaises(ValueError):
                timing.configure('false', directory, 'test')

    def test_spans_pair_on_error_and_keep_thread_context(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = timing.TraceWriter(Path(directory)/'trace.jsonl')
            with patch.object(timing, '_writer', writer):
                def work(step):
                    timing.set_context(step=step)
                    with self.assertRaisesRegex(ValueError, 'test'):
                        with timing.span('operation', rpc_id=str(step)):
                            raise ValueError('test')
                threads = [threading.Thread(target=work,args=(i,)) for i in range(3)]
                for t in threads:t.start()
                for t in threads:t.join()
            writer.close()
            rows = [json.loads(l) for l in writer.path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            for end in [r for r in rows if r['event']=='end']:
                begin=next(r for r in rows if r['id']==end['id'] and r['event']=='begin')
                self.assertEqual(begin['step'],end['step'])
                self.assertEqual(end['rpc_id'],str(end['step']))
                self.assertEqual(end['error'],'ValueError')
                self.assertGreaterEqual(end['duration_ms'],0)

    def test_saturated_queue_drops_instead_of_blocking(self):
        import queue
        writer=timing.TraceWriter.__new__(timing.TraceWriter)
        writer.queue=queue.Queue(maxsize=1)
        writer.queue.put({})
        writer.error=None;writer.dropped=0;writer.stopping=threading.Event()
        writer.emit({'span':'test'})
        self.assertEqual(writer.dropped,1)

    def test_rpc_id_only_added_when_enabled_and_pack_send_recv_preserved(self):
        from expo_ft.env.env_client import EnvClient
        for enabled in [False,True]:
            with tempfile.TemporaryDirectory() as directory:
                writer=timing.TraceWriter(Path(directory)/'trace.jsonl') if enabled else None
                with patch.object(timing,'_writer',writer):
                    client=EnvClient.__new__(EnvClient)
                    client.network_ms=0
                    client._packer=Mock()
                    client._packer.pack.side_effect=lambda x:x
                    conn=Mock(); conn.recv.return_value=b'reply'
                    client._get_connection=Mock(return_value=conn)
                    with patch('expo_ft.env.env_client.msgpack_numpy.unpackb',return_value={'status':'success'}):
                        self.assertEqual(client._call_operation('step',{'action':[0]*7}),{'status':'success'})
                    sent=conn.send.call_args.args[0]
                    self.assertEqual('_timing_id' in sent,enabled)
                    self.assertEqual(sent['action'],[0]*7)
                if writer:
                    writer.close()
                    rows=[json.loads(l) for l in writer.path.read_text().splitlines()]
                    self.assertTrue(all(r['rpc_id']==sent['_timing_id'] for r in rows))
                    self.assertIn('rpc.wait_reply',{r['span'] for r in rows})
