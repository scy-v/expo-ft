"""Shared UR hardware with RLinf's online target bounds and EXPO-owned pacing."""
import numpy as np
from scipy.spatial.transform import Rotation
from .ur import Inference


class OnlineInference(Inference):
    def __init__(self, path):
        super().__init__(path, hardware_only=True)
        self.externally_paced = True
        self.close_threshold = float(self._native_config.get("online_gripper_threshold", 0.5))
        bounds = self._native_config['online_workspace']
        self._lower = np.asarray(bounds['lower'], dtype=float)
        self._upper = np.asarray(bounds['upper'], dtype=float)
        self._anchor = np.asarray(self.init_tcp_pose[3:], dtype=float)
        if self._lower.shape != (6,) or self._upper.shape != (6,) or np.any(self._lower >= self._upper):
            raise ValueError('Expected six ordered online workspace bounds')

    def _target_pose_from_action(self, action, current_tcp_pose=None):
        target = super()._target_pose_from_action(action, current_tcp_pose)
        target[:3] = np.clip(target[:3], self._lower[:3], self._upper[:3])
        euler = Rotation.from_rotvec(target[3:]).as_euler('xyz')
        delta = (euler - self._anchor + np.pi) % (2*np.pi) - np.pi
        delta = np.clip(delta, self._lower[3:] - self._anchor, self._upper[3:] - self._anchor)
        target[3:] = Rotation.from_euler('xyz', self._anchor + delta).as_rotvec()
        return target
