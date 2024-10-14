import numpy as np
from pydrake.all import RigidTransform

import actions
import kinematics
import pzombie


class ErasePolicy:
    def __init__(self):
        self.panda_kinematics = kinematics.Kinematics()
        self.step = 0
        self.curr_policy = None
        self.start_y = None

    def __call__(self, state, time):
        if (self.curr_policy is not None) and (self.curr_policy.done):
            self.step += 1
            self.curr_policy = None
        if self.step == 0 and self.curr_policy is None:
            hover_pose = state["eraser"][4:]
            hover_pose[2] += 0.15
            self.curr_policy = actions.InterpolationPolicy(
                state["panda"][:7], hover_pose, 10.0, 0.075, self.panda_kinematics
            )
        elif self.step == 1 and self.curr_policy is None:
            grasp_pose = state["eraser"][4:]
            grasp_pose[2] += 0.05
            self.curr_policy = actions.InterpolationPolicy(
                state["panda"][:7], grasp_pose, 4.0, 0.075, self.panda_kinematics
            )
        elif self.step == 2 and self.curr_policy is None:
            curr_pose = self.panda_kinematics.fk(state["panda"][:7]).translation()
            self.curr_policy = actions.InterpolationPolicy(
                state["panda"][:7], curr_pose, 1.0, 0.0, self.panda_kinematics
            )
        elif self.step == 3 and self.curr_policy is None:
            whiteboard_hover = self.panda_kinematics.fk(state["panda"][:7])
            whiteboard_hover = whiteboard_hover.multiply(
                RigidTransform([0, 0.00, -0.07])
            ).translation()
            self.curr_policy = actions.InterpolationPolicy(
                state["panda"][:7], whiteboard_hover, 10.0, 0.0, self.panda_kinematics
            )
        elif self.step == 4 and self.curr_policy is None:
            wb_place = state["whiteboard"][4:7]
            wb_place[2] += 0.15
            self.curr_policy = actions.InterpolationPolicy(
                state["panda"][:7], wb_place, 10.0, 0.0, self.panda_kinematics
            )
        elif self.step == 5:
            breakpoint()
            if abs(state["eraser"][5]) < 0.1775:
                if self.start_y is None:
                    self.start_y = state["eraser"][5]
                p_WG_des = self.panda_kinematics.fk(state["panda"]).translation()
                R_WG = self.panda_kinematics.fk(state["panda"]).rotation()
                p_WG_des[1] = self.start_y + (0.01 * (time - 35.0))
                p_WG_des[2] -= 0.05
                desired_qrob = self.panda_kinematics.ik(RigidTransform(p_WG_des, R_WG))
                K = np.array([50.0, 50.0, 100.0, 600.0, 600.0, 100.0])
                return pzombie.AxisAlignedCartesianStiffnessAction(K, desired_qrob, 0.0)
            else:
                return pzombie.PositionAction(state["panda"][:7], 0.0)
        return self.curr_policy(state, time)


def test_simulation():
    eraser = pzombie.Asset(
        "eraser", "assets/eraser.urdf", np.array([1, 0, 0, 0, 0.3, 0.225, 0.725])
    )
    whiteboard = pzombie.Asset(
        "whiteboard", "assets/whiteboard.urdf", np.array([1, 0, 0, 0, 0.2, 0.0, 0.71])
    )
    env = pzombie.Env([eraser, whiteboard])
    drake_env = pzombie.make_env(env)
    pzombie.simulate_policy(ErasePolicy(), drake_env, timeout=40.0)


if __name__ == "__main__":
    test_simulation()
