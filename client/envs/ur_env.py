"""EXPO environment contract over the shared UR evaluation hardware backend."""
import logging
import threading
import time
from pathlib import Path

import yaml
import numpy as np
from scipy.spatial.transform import Rotation

from client.ur.keyboard import EpisodeKeys, KeyboardAction, CheckpointKeys


def to_expo_observation(obs):
    state = np.asarray(obs['observation/state'], dtype=np.float32)
    if state.shape != (19,) or not np.isfinite(state).all():
        raise ValueError('Expected finite UR state19')
    base = np.asarray(obs['observation/image'], dtype=np.uint8)
    wrist = np.asarray(obs['observation/wrist_image'], dtype=np.uint8)
    return {'state': state, 'cartesian_position': state[:6], 'gripper_position': state[18],
            'exterior_image_1_left': base, 'exterior_image_2_left': base,
            'wrist_image_left': wrist, 'prompt': obs['prompt']}


class UREnv:
    def __init__(self, ur_config_path='configs/ur/insert.yaml', replan_steps=None, **kwargs):
        config = yaml.safe_load(Path(ur_config_path).read_text())
        self.reset_control = config['online'].get('reset_teleop', {})
        for name, default in [('hz', 10.0), ('position_step', 0.045), ('rotation_step', 0.075)]:
            value = self.reset_control.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'online.reset_teleop.{name} must be finite and positive')
            self.reset_control[name] = float(value)
        self.manual_checkpoint = config['online'].get('manual_checkpoint', True)
        if type(self.manual_checkpoint) is not bool:
            raise ValueError('online.manual_checkpoint must be a YAML boolean')
        self.checkpoint_keys = CheckpointKeys()
        self.max_episode_chunks = config['online']['max_episode_chunks']
        for name, value in [('max_episode_chunks', self.max_episode_chunks), ('replan_steps', replan_steps)]:
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer; got {value!r}')
        self.episode_action_limit = self.max_episode_chunks * replan_steps
        from client.ur.progress import EpisodeProgress
        self.progress = EpisodeProgress(self.max_episode_chunks, replan_steps)
        from client.ur.online_backend import OnlineInference
        from client.ur.keyboard_listener import KeyboardListener
        self.engine = OnlineInference(Path(ur_config_path))
        self.engine.externally_paced = True
        self.engine.recorder.retain_frames = False  # Online runs must not retain every frame in RAM.
        self.keyboard = None
        self._closed = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self.keys, self.teleop = EpisodeKeys(), KeyboardAction()
        self._outcome = None
        self._started = False
        self._last_obs = None
        self._record_index = 0
        try:
            self.keyboard = KeyboardListener()
            self.engine.connect_robot()
            self.engine.connect_cameras()
            self.engine.connect_gripper()
            self.engine.wait_for_gripper_states()
        except BaseException:
            self.close()
            raise

    def _wait_enter(self, message):
        logging.info(message)
        cutoff = time.time()
        self.keyboard.pop_pressed_events()
        while not self._closed.is_set():
            if any(k in {'Key.enter', 'Key.kpenter'} and t >= cutoff
                   for k, t in self.keyboard.pop_pressed_events()):
                return
            self._closed.wait(0.02)
        raise RuntimeError('UR environment closed while waiting for confirmation')

    def _manual_reset(self):
        """Unrecorded keyboard phase, adapted from RECAP's manual_reset loop."""
        settings = self.reset_control
        teleop = KeyboardAction(position=(settings['position_step'],) * 2,
                               rotation=(settings['rotation_step'],) * 2)
        teleop.gripper = float(self.engine._gripper_position)
        cutoff = time.time()
        self.keyboard.pop_pressed_events()
        blocked = set(self.keyboard.get_pressed_keys())
        logging.info('Manual reset: use movement keys and l/o for gripper; press Right arrow to return home. No rollout data is recorded.')
        try:
            with self._lifecycle_lock:
                if self._closed.is_set():
                    raise RuntimeError('UR environment closed during manual reset')
                self.engine.start_reset_control()
            while not self._closed.is_set():
                tick = time.monotonic()
                events = self.keyboard.pop_pressed_events()
                if any(k == 'Key.right' and t >= cutoff for k, t in events):
                    return
                held = set(self.keyboard.get_pressed_keys())
                if any(k == 'Key.focus_lost' for k, _ in events):
                    blocked.update(held)
                    held = set()
                else:
                    blocked.intersection_update(held)
                    held -= blocked
                with self._lifecycle_lock:
                    if self._closed.is_set():
                        break
                    ee = self.engine.get_ee_pose()
                    action, _ = teleop.action(
                        held, [], np.r_[np.zeros(6), teleop.gripper],
                        ee_rotation=Rotation.from_rotvec(ee[3:]),
                        reference_frame=self.engine.reference_frame,
                        selection=self.engine.select_vector,
                        control_euler=self.engine.control_frame_euler_deg,
                    )
                    self.engine._send_force_action(action)
                    self.engine._gripper_position = float(action[6])
                self._closed.wait(max(0, 1 / settings['hz'] - (time.monotonic() - tick)))
            raise RuntimeError('UR environment closed during manual reset')
        finally:
            with self._lifecycle_lock:
                if not self._closed.is_set():
                    self.engine.stop_force()
            self.keyboard.pop_pressed_events()

    def reset(self):
        self.engine.stop_force()
        if not self._started:
            self._wait_enter('Press Enter to permit the first UR reset.')
        else:
            self._wait_enter('Episode finished. Press Enter to enable keyboard manual reset.')
            self._manual_reset()
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise RuntimeError('UR environment closed before automatic reset')
            self.engine.reset_episode()
        self._wait_enter('UR reset complete. Press Enter to start the episode.')
        self.teleop.reset()
        self.keys.reset()
        if hasattr(self, "checkpoint_keys"):
            self.checkpoint_keys.reset()
        self._outcome = None
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise RuntimeError('UR environment closed during reset')
            self.engine._start_episode()
            self._started = True
            if hasattr(self, "progress"):
                self.progress.update(self.engine._ep_idx, 0)
        return self.get_observation()

    def _events(self):
        events = self.keyboard.pop_pressed_events()
        if self._started and self._outcome is None and hasattr(self, 'checkpoint_keys'):
            was_requested = self.checkpoint_keys.requested
            self.checkpoint_keys.consume(events)
            if self.manual_checkpoint and self.checkpoint_keys.requested and not was_requested:
                logging.info('Checkpoint requested: save after this episode ends.')
        for key, _ in events:
            if key == 'Key.slash':
                self.teleop.mode = 1 - self.teleop.mode
        if self._outcome is None:
            self.keys.consume(events)
            outcome = self.keys.outcome
            if outcome is None and self.engine._ep_steps >= self.episode_action_limit:
                outcome = False
            if outcome is not None:
                self._outcome = outcome
                if hasattr(self, "progress"):
                    self.progress.close()
                self.engine._submit_episode('manual_success' if outcome else 'failure_or_timeout', outcome)
                self.engine._ep_idx += 1
        return events

    def get_observation(self):
        self._events()
        raw = self.engine.get_obs_state()
        self._last_obs = to_expo_observation(raw)
        self.engine.recorder.submit_obs(raw)
        return self._last_obs

    def get_checkpoint_control(self):
        return {'manual': self.manual_checkpoint, 'requested': self.checkpoint_keys.requested}

    def get_info_for_step(self):
        # The network wrapper attaches these labels to the same observation.
        done = self._outcome is not None
        return done, bool(self._outcome), float(bool(self._outcome)), float(not done)

    def step(self, action, *, action_available=True):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Expected seven finite UR delta/gripper commands')
        events = self._events()
        if self._outcome is not None:
            # EXPO can send a step after seeing done; never move after a terminal label.
            return {'executed_action': np.r_[np.zeros(6), self.engine._gripper_position],
                    'action_type': 'policy', 'executed': False}
        if not action_available:
            # EXPO polls for takeover release with no queued policy action. A zero
            # DROID velocity is idle; a zero UR gripper position would close it.
            action = np.r_[np.zeros(6), self.engine._gripper_position]
        ee = self.engine.get_ee_pose()
        actual, human = self.teleop.action(
            self.keyboard.get_pressed_keys(), [], action,
            ee_rotation=Rotation.from_rotvec(ee[3:]),
            reference_frame=self.engine.reference_frame, selection=self.engine.select_vector,
            control_euler=self.engine.control_frame_euler_deg,
        )
        if not human and not action_available:
            return {'executed_action': action, 'action_type': 'policy', 'executed': False}
        self.engine._send_force_action(actual)
        self.engine._gripper_position = float(actual[6])
        self.engine._ep_steps += 1
        if hasattr(self, "progress"):
            self.progress.update(self.engine._ep_idx, self.engine._ep_steps)
        self._record_index += 1
        self.engine.recorder.submit_actions(
            actual[None], self._record_index, self.engine.task_description,
            state=self._last_obs['state'], episode_index=self.engine._ep_idx,
            episode_action_count=self.engine._ep_steps, action_horizon=1,
            controller='human' if human else 'expo',
        )
        return {'executed_action': actual, 'action_type': 'human' if human else 'policy'}

    def render(self):
        return self.engine.get_obs_state()['observation/image']

    def close(self):
        self._closed.set()
        if hasattr(self, "progress"):
            self.progress.close()
        with self._lifecycle_lock:
            try:
                self.engine.close()
            finally:
                try:
                    if self.keyboard is not None:
                        self.keyboard.close()
                finally:
                    self.engine.recorder.close()
