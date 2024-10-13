import numpy as np
from pydrake.all import RigidTransform

import kinematics
import pzombie


class ExamplePolicy:
    def __init__(self, start_pose):
        self.panda_kinematics = kinematics.Kinematics()
        self.start_pose = start_pose

    def __call__(self, state, time):
        q = state["panda"][:7]
        X_BG = self.panda_kinematics.fk(q)
        offset = RigidTransform()
        X_BG_des = X_BG.multiply(offset)
        q_des = self.panda_kinematics.ik(X_BG_des)
        return pzombie.PositionAction(q_des, 0.02)


def test_simulation():
    eraser = pzombie.Asset(
        "eraser", "assets/eraser.urdf", np.array([1, 0, 0, 0, 0.6, 0.0, 0.725])
    )
    env = pzombie.Env([eraser])
    drake_env = pzombie.make_env(env)
    pzombie.simulate_policy(ExamplePolicy(None), drake_env)


if __name__ == "__main__":
    test_simulation()
