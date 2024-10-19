import curses
from curses import wrapper

import numpy as np
from pydrake.all import RigidTransform

import actions
import kinematics
import pzombie


class TeleopPolicy:
    def __init__(self, stdscr):
        self.panda = kinematics.Kinematics()
        self.last_called = -1.0
        self.last_called_gripper = -1.0
        self.sp = None
        self.gripper = 1.0
        self.K = np.diag(np.array([30.0, 30.0, 30.0, 200.0, 200.0, 200.0]))
        self.key_buf = []
        self.stdscr = stdscr

    def handle_keypress(self, k, t):
        if k == ord(" ") and (t - self.last_called_gripper) > 1.0:
            self.last_called_gripper = t
            self.gripper = 1 - self.gripper
        elif k != -1:
            self.key_buf.append(k)

    def update_sp(self, X_WG):
        sp_SE3 = X_WG
        if ord("w") in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([0.01, 0, 0]))
        if ord("s") in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([-0.01, 0, 0]))
        if ord("a") in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([0, -0.01, 0]))
        if ord("d") in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([0, 0.01, 0]))
        if curses.KEY_UP in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([0, 0, -0.01]))
        if curses.KEY_DOWN in self.key_buf:
            sp_SE3 = sp_SE3.multiply(RigidTransform([0, 0, 0.01]))
        self.sp = self.panda.ik(sp_SE3)
        self.key_buf = []

    def __call__(self, state, time):
        k = self.stdscr.getch()
        self.handle_keypress(k, time)
        if self.sp is None:
            self.sp = state["panda"][:7]
        if (time - self.last_called) >= 0.01:
            self.update_sp(self.panda.fk(self.sp))
            self.last_called = time
        return actions.CartesianStiffnessAction(self.K, self.sp, self.gripper)


def test_simulation(stdscr):
    stdscr.nodelay(1)
    eraser = pzombie.Asset(
        "eraser", "assets/eraser.urdf", np.array([1, 0, 0, 0, 0.3, 0.225, 0.725])
    )
    whiteboard = pzombie.Asset(
        "whiteboard", "assets/whiteboard.urdf", np.array([1, 0, 0, 0, 0.2, 0.0, 0.71])
    )
    env = pzombie.Env([eraser, whiteboard])
    drake_env = pzombie.make_env(env)
    pzombie.simulate_policy(TeleopPolicy(stdscr), drake_env, timeout=60.0)


if __name__ == "__main__":
    wrapper(test_simulation)
