import numpy as np
from pydrake.all import RigidTransform

from pzombie import actions, components, kinematics, pzombie


#  given the intiial pose of the eraser and whiteboard,
#  return a sequence of policies to grasp the eraser
#  and press the whiteboard
def compute_plan(q0, panda):
    eraser = q0["eraser"][4:7]
    wb = q0["whiteboard"][4:7]
    q_eraserhover = eraser + np.array([0, 0, 0.15])
    pi_eraserhover = actions.InterpolationPolicy(
        q_eraserhover, 10.0, components.GRIPPER_OPEN, panda
    )
    q_erasergrasp = eraser + np.array([0, 0, 0.05])
    pi_erasergrasp = actions.InterpolationPolicy(
        q_erasergrasp, 4.0, components.GRIPPER_OPEN, panda
    )
    pi_grasp = actions.GripperPolicy("close")
    q_eraserlift = eraser + np.array([0.0, 0.0, 0.16])
    pi_eraserlift = actions.InterpolationPolicy(
        q_eraserlift, 5.0, components.GRIPPER_CLOSE, panda
    )
    q_place = wb + np.array([0.0, 0.0, 0.14])
    pi_place = actions.InterpolationPolicy(
        q_place, 10.0, components.GRIPPER_CLOSE, panda
    )
    return [pi_eraserhover, pi_erasergrasp, pi_grasp, pi_eraserlift, pi_place]


def compute_wipe_policy(panda, state):
    K = np.diag(np.array([10.0, 10.0, 100.0, 600.0, 600.0, 50.0]))
    R_WG_0 = panda.fk(state["panda"][:7]).rotation()
    p_WG_0 = panda.fk(state["panda"][:7]).translation() - np.array([0, 0, 0.05])
    X_WG_0 = RigidTransform(R_WG_0, p_WG_0)
    p_WG_des = np.array([p_WG_0[0], 0.175, p_WG_0[2]])
    return actions.InterpolationPolicy(
        p_WG_des, 10.0, components.GRIPPER_CLOSE, panda, K=K, start=X_WG_0
    )


class FullErasePolicy:
    def __init__(self):
        self.panda_kinematics = kinematics.Kinematics()
        self.step = 0
        self.place_plan = None
        self.wipe_policy = None

    def __call__(self, state, time):
        # one-time call to compute place_plan which is an
        # "open-loop" sequence of motions to pick up eraser and put it on the wb
        if self.place_plan is None:
            self.place_plan = compute_plan(state, self.panda_kinematics)
        # if current step in place_plan is completed, move to next step (sub-policy)
        if self.step < len(self.place_plan) and self.place_plan[self.step].done:
            self.step += 1
        # if still placing, pass state to current sub-policy in place plan
        if self.step < len(self.place_plan):
            pi_curr = self.place_plan[self.step]
            return pi_curr(state, time)
        # if we are done placing, we are wiping -- call that policy
        else:
            if self.wipe_policy is None:
                self.wipe_policy = compute_wipe_policy(self.panda_kinematics, state)
            return self.wipe_policy(state, time)


def test_simulation():
    eraser = components.Asset(
        "eraser", "assets/eraser.urdf", np.array([1, 0, 0, 0, 0.3, 0.225, 0.725])
    )
    whiteboard = components.Asset(
        "whiteboard", "assets/whiteboard.urdf", np.array([1, 0, 0, 0, 0.2, 0.0, 0.71])
    )
    env = components.Env([eraser, whiteboard])
    pzombie.simulate_policy(FullErasePolicy(), env, timeout=60.0)


if __name__ == "__main__":
    test_simulation()
