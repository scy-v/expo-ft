"""CPU regression tests for UR action semantics and EXPO environment boundaries."""
import asyncio
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from client.ur.keyboard import KeyboardAction, EpisodeKeys
from client.envs.ur_env import UREnv, to_expo_observation
from client.ur.policy import validate_config, example_observation, URPolicy
from client.ur.ur import Inference, _ContinuousForceController


class KeyboardTests(unittest.TestCase):
    def test_control_frame_converts_to_tcp_and_release_restores_policy(self):
        keyboard = KeyboardAction()
        rotation = Rotation.from_euler('z', 90, degrees=True)
        kwargs = dict(ee_rotation=rotation, reference_frame='tcp', selection=[1,1,1,0,0,1])
        policy = np.array([0.1,0.2,0.3,0,0,0,1])
        action, active = keyboard.action({'s','r'}, [], policy, **kwargs)
        self.assertTrue(active)
        np.testing.assert_allclose(action[:3], [0,-0.03,0], atol=1e-8)
        np.testing.assert_allclose(action[3:6], 0, atol=1e-8)
        action, active = keyboard.action(set(), [], policy, **kwargs)
        self.assertFalse(active)
        np.testing.assert_array_equal(action, policy)
        action, active = keyboard.action({'l'}, [], policy, **kwargs)
        self.assertTrue(active)
        self.assertEqual(action[6], 0)
        action, active = keyboard.action({'s','Key.leftbrace'}, [], policy, **kwargs)
        self.assertTrue(active)
        np.testing.assert_array_equal(action[:6], 0)

    def test_fine_mode_and_key_precedence(self):
        keyboard = KeyboardAction()
        kw = dict(ee_rotation=Rotation.identity(), reference_frame='base', selection=np.ones(6))
        action, _ = keyboard.action({'s','w'}, ['Key.slash'], np.zeros(7), **kw)
        self.assertAlmostEqual(action[0], -0.006)
        keyboard.reset()
        action, _ = keyboard.action({'b'}, [], np.zeros(7), **kw)
        self.assertAlmostEqual(action[5], 0.05)

    def test_episode_edges_and_reset_do_not_leak(self):
        keys = EpisodeKeys()
        self.assertIsNone(keys.consume([('Key.space',1),('Key.space',2)]))
        self.assertTrue(keys.consume([('Key.space',2.2)]))
        keys.reset()
        self.assertIsNone(keys.consume([('Key.enter',3)]))
        self.assertFalse(keys.consume([('Key.enter',3.3)]))
        keys.reset()
        self.assertIsNone(keys.consume([('Key.enter',4),('Key.focus_lost',4.1),('Key.enter',4.2)]))


class EnvTests(unittest.TestCase):
    def env(self):
        env = UREnv.__new__(UREnv)
        env.keys, env.teleop = EpisodeKeys(), KeyboardAction()
        env._outcome = None
        env._started = True
        env._closed = threading.Event()
        env._lifecycle_lock = threading.RLock()
        env._last_obs = to_expo_observation(example_observation())
        env._record_index = 0
        env.keyboard = Mock()
        env.keyboard.pop_pressed_events.return_value=[]
        env.keyboard.get_pressed_keys.return_value=set()
        env.engine=Mock()
        env.engine._ep_start=time.perf_counter()
        env.engine.ep_timeout=60
        env.engine._ep_idx=1
        env.engine._ep_steps=0
        env.engine._gripper_position=1.
        env.engine.get_ee_pose.return_value=np.zeros(6)
        env.engine.reference_frame='tcp'
        env.engine.select_vector=[1,1,1,0,0,1]
        env.engine.control_frame_euler_deg=[0,0,0]
        env.engine.task_description='insert'
        env.engine.get_obs_state.return_value=example_observation()
        return env

    def test_close_interrupts_reset_confirmation_and_releases_resources(self):
        env = self.env()
        env._started = False
        with patch.object(env, '_wait_enter', side_effect=RuntimeError('closed')):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                env.reset()
        env.engine._start_episode.assert_not_called()
        env.engine.close.side_effect = RuntimeError('controller cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            env.close()
        self.assertTrue(env._closed.is_set())
        env.keyboard.close.assert_called_once()
        env.engine.recorder.close.assert_called_once()

    def test_real_action_is_recorded_and_no_extra_pacing(self):
        env=self.env()
        env.keyboard.get_pressed_keys.return_value={'s'}
        with patch('time.sleep',side_effect=AssertionError('online env must not pace')):
            response=env.step(np.zeros(7))
        self.assertEqual(response['action_type'],'human')
        np.testing.assert_array_equal(env.engine._send_force_action.call_args.args[0],response['executed_action'])
        self.assertAlmostEqual(response['executed_action'][0],0.03)
        env.keyboard.get_pressed_keys.return_value=set()
        self.assertEqual(env.step(np.zeros(7))['action_type'],'policy')

    def test_terminal_is_latched_and_never_executes_after_success(self):
        env=self.env()
        env.keyboard.pop_pressed_events.return_value=[('Key.space',1),('Key.space',1.2)]
        env.get_observation()
        self.assertEqual(env.get_info_for_step(),(True,True,1.,0.))
        self.assertFalse(env.step(np.zeros(7))['executed'])
        env.engine._send_force_action.assert_not_called()
        env.engine._submit_episode.assert_called_once()

    def test_mode_toggle_consumed_by_observation_is_preserved(self):
        env=self.env()
        env.keyboard.pop_pressed_events.side_effect=[[('Key.slash',1)],[]]
        env.get_observation()
        env.keyboard.get_pressed_keys.return_value={'s'}
        self.assertAlmostEqual(env.step(np.zeros(7))['executed_action'][0],0.006)

    def test_takeover_release_does_not_close_position_gripper(self):
        env=self.env()
        env.engine._gripper_position=1.
        result=env.step(np.zeros(7),action_available=False)
        self.assertEqual(result['action_type'],'policy')
        self.assertFalse(result['executed'])
        self.assertEqual(result['executed_action'][6],1.)
        env.engine._send_force_action.assert_not_called()
        env.keyboard.get_pressed_keys.return_value={'s'}
        result=env.step(np.zeros(7),action_available=False)
        self.assertEqual(result['action_type'],'human')
        self.assertEqual(result['executed_action'][6],1.)

    def test_state_and_image_contract(self):
        raw=example_observation()
        raw['observation/state']=np.arange(19,dtype=np.float32)
        converted=to_expo_observation(raw)
        np.testing.assert_array_equal(converted['state'],raw['observation/state'])
        self.assertEqual(converted['gripper_position'],18)
        self.assertEqual(converted['exterior_image_1_left'].shape,(224,224,3))
        raw['observation/state']=np.zeros(7)
        with self.assertRaises(ValueError): to_expo_observation(raw)

    def test_external_pacing_bypasses_standalone_sleep(self):
        engine=Inference.__new__(Inference)
        engine.externally_paced=True
        with patch('time.sleep',side_effect=AssertionError('double pacing')):
            engine._pace_action_target()


class ControlTests(unittest.TestCase):
    def test_impedance_matches_physical_units(self):
        e=Inference.__new__(Inference)
        e.pos_delta=.048; e.vel_delta=4.;e.kp=1200;e.kd=600;e.kp_rot=20;e.kd_rot=4
        e.use_control_frame=False
        force=e._calculate_force(np.array([.1,0,0,0,0,.2]),np.zeros(6),np.array([.01,0,0,0,0,.1]))
        np.testing.assert_allclose(force,[51.6,0,0,0,0,3.6],atol=1e-8)

    def test_background_controller_tracks_latest_absolute_target(self):
        engine=Mock()
        engine.rtde_freq=100
        engine.rtde_r.getActualTCPPose.return_value=np.zeros(6)
        engine.rtde_r.getActualTCPSpeed.return_value=np.zeros(6)
        engine._calculate_force.return_value=np.zeros(6)
        engine.rtde_c.forceMode.return_value=True
        engine.task_frame=[0]*6;engine.select_vector=[1]*6;engine.force_type=2;engine.force_limit=[2]*6
        control=_ContinuousForceController(engine)
        control.start()
        try:
            target=np.array([.1,0,0,0,0,0])
            control.update_target(target)
            time.sleep(.04)
            self.assertGreaterEqual(engine.rtde_c.forceMode.call_count,3)
            np.testing.assert_array_equal(control._target,target)
        finally:
            control.stop()


class ConfigTests(unittest.TestCase):
    def test_default_config_and_actual_norm_stats(self):
        cfg=yaml.safe_load(Path('configs/ur/insert.yaml').read_text())
        validate_config(cfg,files=False)
        cfg['robot']['action_fps']=20
        with self.assertRaises(ValueError):validate_config(cfg,files=False)

    def test_sft_adapter_and_expo_adapter_share_ur_output(self):
        for kind in ['sft','expo']:
            policy=URPolicy.__new__(URPolicy)
            policy.kind=kind;policy.steps=8
            expected=np.arange(56).reshape(8,7)
            if kind=='sft':
                policy.policy=Mock()
                policy.policy.infer.return_value={'actions':expected[None]}
            else:
                policy.agent=Mock()
                policy.agent.sample_actions.return_value=(expected,policy.agent,{})
            result=policy.infer(example_observation())
            np.testing.assert_array_equal(result['actions'],expected)
            self.assertEqual(result['controller'],kind)



class WorkspaceTests(unittest.TestCase):
    def test_online_clipping_preserves_continuous_euler_window(self):
        from client.ur.online_backend import OnlineInference
        engine=OnlineInference.__new__(OnlineInference)
        engine._lower=np.array([0,-.45,0,-3.66,-.52,-3.14])
        engine._upper=np.array([.9,.45,.6,-2.61,.52,0])
        engine._anchor=np.array([-3.14,0,-1.57])
        raw=np.r_[2.,1.,1.,Rotation.from_euler('xyz',[3.14,0,-1.57]).as_rotvec()]
        with patch.object(Inference,'_target_pose_from_action',return_value=raw):
            target=engine._target_pose_from_action(np.zeros(7))
        np.testing.assert_allclose(target[:3],[.9,.45,.6])
        np.testing.assert_allclose(Rotation.from_rotvec(target[3:]).as_matrix(),
                                   Rotation.from_euler('xyz',[3.14,0,-1.57]).as_matrix(),atol=1e-7)

if __name__ == "__main__":
    unittest.main()
