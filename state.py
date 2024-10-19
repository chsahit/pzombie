from dataclasses import dataclass

import numpy as np


class State:
    def __init__(self, state_dict, contact_state):
        self._state = state_dict
        self.contact_state = contact_state

    def __getitem__(self, idx):
        return self._state[idx]


@dataclass(frozen=True)
class Contact:
    body_A: str
    body_B: str
    centroid: np.ndarray
    F_Ac_W: np.ndarray
