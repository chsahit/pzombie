from dataclasses import dataclass
from typing import List

import numpy as np

DEFAULT_PANDA_ANGLES = np.array(
    [
        0.0796904,
        0.18628879,
        -0.07548908,
        -2.42085905,
        0.06961755,
        2.52396334,
        0.6796144,
        0.03,
        0.03,
    ]
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0


class Asset:
    def __init__(self, name, path, pose):
        self.name = name
        self.path = path
        self.pose = pose


class Env:
    def __init__(
        self, assets: List[Asset], panda_pose: np.ndarray = DEFAULT_PANDA_ANGLES
    ):
        self.panda_pose = panda_pose
        self.assets = assets


class State:
    def __init__(self, state_dict, contact_state, jacobian):
        self._state = state_dict
        self.contact_state = contact_state
        self.jacobian = jacobian

    def __getitem__(self, idx):
        return self._state[idx]


@dataclass(frozen=True)
class Contact:
    body_A: str
    body_B: str
    centroid: np.ndarray
    F_Ac_W: np.ndarray
