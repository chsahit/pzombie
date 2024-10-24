import curses
from curses import wrapper

import numpy as np
from pydrake.all import RigidTransform

from pzombie import actions, components, kinematics, pzombie


class TeleopPolicy:
    def __init__(self, stdscr):
        self.panda = kinematics.Kinematics()
        self.last_called = -1.0
        self.last_called_gripper = -1.0
        self.sp_R7 = None
        self.sp_SE3 = None
        self.gripper = 1.0
        self.K = np.diag(np.array([40.0, 40.0, 100.0, 200.0, 200.0, 200.0]))
        # self.K = np.diag(np.array([100.0, 100.0, 100.0, 600.0, 600.0, 600.0]))
        self.key_buf = []
        self.stdscr = stdscr

    def handle_keypress(self, k, t):
        if k == ord(" ") and (t - self.last_called_gripper) > 1.0:
            self.last_called_gripper = t
            self.gripper = 1 - self.gripper
        elif k != -1 and (k not in self.key_buf):
            self.key_buf.append(k)

    def update_sp(self, panda_joints):
        if ord("w") in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([0.01, 0, 0]))
        if ord("s") in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([-0.01, 0, 0]))
        if ord("a") in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([0, -0.01, 0]))
        if ord("d") in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([0, 0.01, 0]))
        if curses.KEY_UP in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([0, 0, -0.01]))
        if curses.KEY_DOWN in self.key_buf:
            self.sp_SE3 = self.sp_SE3.multiply(RigidTransform([0, 0, 0.01]))
        self.sp_R7 = self.panda.ik(self.sp_SE3, joint_center=panda_joints)
        self.key_buf = []

    def __call__(self, state, time):
        k = self.stdscr.getch()
        self.handle_keypress(k, time)
        if self.sp_R7 is None:
            self.sp_R7 = state["panda"][:7]
            self.sp_SE3 = self.panda.fk(self.sp_R7)
        if (time - self.last_called) >= 0.01:
            self.update_sp(state["panda"][:7])
            self.last_called = time
        return actions.CartesianStiffnessAction(self.K, self.sp_R7, self.gripper)


def test_simulation(stdscr):
    stdscr.nodelay(1)
    eraser = components.Asset(
        "eraser", "assets/eraser.urdf", np.array([1, 0, 0, 0, 0.3, 0.225, 0.725])
    )
    whiteboard = components.Asset(
        "whiteboard", "assets/whiteboard.urdf", np.array([1, 0, 0, 0, 0.2, 0.0, 0.71])
    )
    env = components.Env([eraser, whiteboard])
    pzombie.simulate_policy(
        TeleopPolicy(stdscr), env, timeout=300.0, target_rtr=1.0
    )


if __name__ == "__main__":
    wrapper(test_simulation)
