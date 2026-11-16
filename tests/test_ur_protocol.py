"""MessagePack client protocol tests using a fake UR, never opening robot sockets."""
import asyncio
import threading
import unittest
from unittest.mock import patch, Mock

import numpy as np
from ml_collections import ConfigDict
from openpi_client import msgpack_numpy
import websockets.exceptions

from client import run_client
from client.ur.policy import example_observation
from client.envs.ur_env import to_expo_observation


class FakeUR:
    instances=[]
    def __init__(self, **kwargs):
        self.replan_steps = kwargs.get("replan_steps")
        self.closed=False
        self.terminal=False
        self.instances.append(self)
    def reset(self):return to_expo_observation(example_observation())
    def get_observation(self):return self.reset()
    def get_info_for_step(self):return self.terminal,self.terminal,float(self.terminal),float(not self.terminal)
    def step(self, action, **kwargs):
        self.last_available = kwargs.get('action_available', True)
        self.terminal=bool(action[0] == 9)
        return {'executed_action':np.ones(7)*.1,'action_type':'human', 'executed':not self.terminal}
    def close(self):self.closed=True


class Wire:
    remote_address='fake'
    transport=Mock()
    def __init__(self, requests):
        self.requests=iter(requests)
        self.responses=[]
        self.codec=msgpack_numpy.Packer()
        self.closed = asyncio.Event()
    async def recv(self):
        try:return self.codec.pack(next(self.requests))
        except StopIteration:raise websockets.exceptions.ConnectionClosedOK(None,None)
    async def send(self, data):self.responses.append(msgpack_numpy.unpackb(data))
    async def wait_closed(self):await self.closed.wait()


class ProtocolTests(unittest.TestCase):
    def test_cancelling_pending_reset_releases_worker(self):
        class PendingReset:
            def __init__(self):
                self.entered = threading.Event()
                self.closed = threading.Event()
            def reset(self):
                self.entered.set()
                self.interrupted = self.closed.wait(1.0)
            def close(self):
                self.closed.set()
        env = PendingReset()
        wire = Wire([])
        async def scenario():
            task = asyncio.create_task(run_client._reset_ur_environment(env, wire))
            self.assertTrue(await asyncio.to_thread(env.entered.wait, 1.0))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(scenario())
        self.assertTrue(env.interrupted)
        self.assertTrue(env.closed.is_set())

    def test_invalid_ur_action_ends_session_before_next_request(self):
        FakeUR.instances = []
        task = ConfigDict(dict(env=FakeUR, env_name='test', env_type='ur', language_instruction='insert'))
        wire = Wire([{'operation': 'create_env', 'env_usage': 'train'},
                     {'operation': 'step', 'env_id': 'test_train_1', 'action': [float('nan')]*7},
                     {'operation': 'get_observation', 'env_id': 'test_train_1'}])
        run_client._env_storage.clear()
        run_client._env_create_counter = 0
        with patch.object(run_client, 'load_task_config', return_value=task):
            asyncio.run(run_client._handle_environment_request(wire))
        self.assertEqual(len(wire.responses), 2)
        self.assertEqual(wire.responses[-1]['status'], 'error')
        self.assertTrue(FakeUR.instances[0].closed)
        self.assertFalse(run_client._env_storage)

    def test_disconnect_interrupts_reset_wait_without_blocking_event_loop(self):
        class WaitingUR(FakeUR):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.release = threading.Event()
            def reset(self):
                self.interrupted = self.release.wait(0.5)
                return to_expo_observation(example_observation())
            def close(self):
                super().close()
                self.release.set()
        FakeUR.instances = []
        task = ConfigDict(dict(env=WaitingUR, env_name='wait', env_type='ur', language_instruction='insert'))
        wire = Wire([{'operation': 'create_env', 'env_usage': 'train'},
                     {'operation': 'reset', 'env_id': 'wait_train_1'}])
        run_client._env_storage.clear()
        run_client._env_create_counter = 0
        async def scenario():
            asyncio.get_running_loop().call_later(0.02, wire.closed.set)
            await run_client._handle_environment_request(wire)
        with patch.object(run_client, 'load_task_config', return_value=task):
            asyncio.run(scenario())
        self.assertTrue(FakeUR.instances[0].interrupted)
        self.assertTrue(FakeUR.instances[0].closed)
        self.assertFalse(run_client._env_storage)
        self.assertFalse(any('observation' in r for r in wire.responses))

    def test_ur_override_execution_status_labels_and_disconnect_cleanup(self):
        FakeUR.instances=[]
        task=ConfigDict(dict(env=FakeUR,env_name='test',env_type='ur',language_instruction='insert'))
        wire=Wire([{'operation':'create_env','env_usage':'train'},
            {'operation':'step','env_id':'test_train_1','action':np.zeros(7)},
            {'operation':'step','env_id':'test_train_1','action':np.array([9,0,0,0,0,0,0])},
            {'operation':'get_observation','env_id':'test_train_1'}])
        run_client._env_storage.clear()
        run_client._env_create_counter=0
        with patch.object(run_client,'load_task_config',return_value=task):
            asyncio.run(run_client._handle_environment_request(wire))
        self.assertEqual(len(wire.responses),4)
        self.assertEqual(wire.responses[1]['action_type'],'human')
        self.assertTrue(wire.responses[1]['executed'])
        self.assertFalse(wire.responses[2]['executed'])
        self.assertEqual(wire.responses[3]['reward'],1.)
        self.assertEqual(wire.responses[3]['mask'],0.)
        np.testing.assert_array_equal(wire.responses[3]['observation']['state'],np.zeros(19))
        self.assertTrue(FakeUR.instances[0].closed)
        self.assertFalse(run_client._env_storage)



class LoopbackTests(unittest.TestCase):
    def test_real_websocket_disconnect_releases_pending_reset(self):
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        entered = threading.Event()
        released = threading.Event()
        class WaitingUR(FakeUR):
            def reset(self):
                entered.set()
                self.interrupted = released.wait(1.0)
                return to_expo_observation(example_observation())
            def close(self):
                super().close()
                released.set()
        FakeUR.instances = []
        run_client._env_storage.clear()
        run_client._env_create_counter = 0
        task = ConfigDict(dict(env=WaitingUR, env_name='wait', env_type='ur', language_instruction='insert'))
        failures = []
        async def learner(connection):
            try:
                codec = msgpack_numpy.Packer()
                await connection.send(codec.pack(dict(operation='create_env', env_usage='train')))
                response = msgpack_numpy.unpackb(await connection.recv())
                await connection.send(codec.pack(dict(operation='reset', env_id=response['env_id'])))
                self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))
                await connection.close()
            except Exception as exc:
                failures.append(exc)
        async def scenario():
            async with serve(learner, '127.0.0.1', 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f'ws://127.0.0.1:{port}') as connection:
                    await asyncio.wait_for(run_client._handle_environment_request(connection), 3)
        with patch.object(run_client, 'load_task_config', return_value=task):
            asyncio.run(scenario())
        self.assertFalse(failures)
        self.assertTrue(FakeUR.instances[-1].interrupted)
        self.assertTrue(FakeUR.instances[-1].closed)
        self.assertFalse(run_client._env_storage)

    def test_real_websocket_environment_rpc(self):
        import threading
        import websockets.asyncio.client
        from expo_ft.env.env_client import EnvClient
        task=ConfigDict(dict(env=FakeUR,env_name='wire',env_type='ur',language_instruction='insert'))
        learner=EnvClient(host='127.0.0.1',port=0)
        learner._ensure_server()
        port=learner._server.socket.getsockname()[1]
        failures=[]
        async def robot():
            async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}',compression=None,max_size=None) as conn:
                await run_client._handle_environment_request(conn)
        def target():
            try:asyncio.run(robot())
            except Exception as exc:failures.append(exc)
        with patch.object(run_client,'load_task_config',return_value=task):
            thread=threading.Thread(target=target,daemon=True)
            thread.start()
            try:
                env_id,prompt=learner.create_env({'env_usage':'train', 'replan_steps':3})
                self.assertEqual(FakeUR.instances[-1].replan_steps, 3)
                learner.reset(env_id)
                action,kind=learner.step(env_id,np.zeros(7))
                self.assertEqual(kind,'human')
                np.testing.assert_allclose(action,.1)
                self.assertTrue(learner.last_step_executed)
                learner.step(env_id,np.array([9,0,0,0,0,0,0]),action_available=False)
                self.assertFalse(FakeUR.instances[-1].last_available)
                self.assertFalse(learner.last_step_executed)
                obs=learner.get_observation(env_id)
                self.assertEqual(obs['state'].shape,(19,))
                self.assertEqual(learner.get_info_for_step(env_id),(True,True,1.,0.))
            finally:
                learner.close()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(failures)


if __name__ == "__main__":
    unittest.main()
