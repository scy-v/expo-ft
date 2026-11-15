# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.RLinf.
"""UR keyboard takeover; mapping and control-frame conversion from RLinf f54ac81."""
import numpy as np
from scipy.spatial.transform import Rotation


class KeyboardAction:
    # Same precedence as the collection/RLinf keyboard implementation.
    KEYS = {'s': (0, 1), 'w': (0, -1), 'd': (1, 1), 'a': (1, -1),
            'q': (2, 1), 'e': (2, -1), 'r': (3, 1), 't': (3, -1),
            'g': (4, 1), 'f': (4, -1), 'b': (5, 1), 'v': (5, -1)}

    def __init__(self, position=(0.03, 0.006), rotation=(0.05, 0.03)):
        self.position, self.rotation = position, rotation
        self.mode, self.gripper = 0, 1.0

    def reset(self):
        self.mode, self.gripper = 0, 1.0

    def action(self, held, presses, policy_action, *, ee_rotation, reference_frame,
               selection, control_euler=(0, 0, 0)):
        if 'Key.slash' in presses:
            self.mode = 1 - self.mode
        pose = np.zeros(6)
        for key, (axis, direction) in self.KEYS.items():
            if key in held:
                pose[axis] = direction * (self.position[self.mode] if axis < 3 else self.rotation[self.mode])
        active = bool(np.any(pose) or held & {'l', 'o', 'Key.leftbrace'})
        if not active:
            self.gripper = float(policy_action[-1] > 0.5)
            return np.asarray(policy_action), False
        if 'Key.leftbrace' in held:
            pose[:] = 0
        if 'l' in held:
            self.gripper = 0.0
        elif 'o' in held:
            self.gripper = 1.0
        pose *= np.asarray(selection)
        control = Rotation.from_euler('xyz', control_euler, degrees=True)
        xyz = control.apply(pose[:3])
        delta = control * Rotation.from_euler('xyz', pose[3:]) * control.inv()
        if reference_frame == 'tcp':
            xyz = ee_rotation.inv().apply(xyz)
            delta = ee_rotation.inv() * delta * ee_rotation
        elif reference_frame != 'base':
            raise ValueError(reference_frame)
        return np.r_[xyz, delta.as_euler('xyz'), self.gripper], True


class EpisodeKeys:
    """Edge-triggered success/failure labels; held-key repeat never ends an episode."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.reset()

    def reset(self):
        self.first = {}
        self.outcome = None

    def consume(self, events):
        for key, stamp in events:
            if key == 'Key.focus_lost':
                self.first.clear()
                continue
            key = 'Key.enter' if key == 'Key.kpenter' else key
            if key not in {'Key.enter', 'Key.space'}:
                continue
            previous = self.first.get(key)
            self.first[key] = stamp
            if previous is not None and 0 <= stamp - previous <= self.interval:
                self.outcome = key == 'Key.space'
                self.first.clear()
                break
        return self.outcome
