import numpy as np
from pydrake.all import RigidTransform


class CartesianStiffnessAction:
    def __init__(self, stiffness, joint_angles, gripper_width):
        self.stiffness = stiffness
        self.joint_angles = joint_angles
        self.gripper_width = gripper_width

    def serialize(self):
        qdot = np.zeros((9,))
        gripper_q = self.gripper_width / 2.0
        q = np.concatenate([self.joint_angles, np.array([gripper_q, gripper_q])])
        x_des = np.concatenate([q, qdot])
        x_d_K = np.concatenate([self.stiffness.flatten(), x_des])
        return x_d_K


class AxisAlignedCartesianStiffnessAction(CartesianStiffnessAction):
    def __init__(self, stiffness, joint_angles, gripper_width):
        stiffness = np.diag(stiffness)
        super().__init__(stiffness, joint_angles, gripper_width)


class PositionAction(AxisAlignedCartesianStiffnessAction):
    def __init__(self, joint_angles, gripper_width):
        stiffness = np.array([100, 100, 100, 600, 600, 600])
        super().__init__(stiffness, joint_angles, gripper_width)


class InterpolationPolicy:
    def __init__(self, qrob, goal_pose, total_time, grasp, panda):
        self.start_time = None
        self.done = None
        self.total_time = total_time
        self.grasp = grasp
        # waypoints
        num_waypoints = 10
        self.time_per_waypoint = float(total_time) / num_waypoints
        R_WG = panda.fk(qrob).rotation()
        curr_p_WG = panda.fk(qrob).translation()
        vel = (1.0 / num_waypoints) * (goal_pose - curr_p_WG)
        pts = [curr_p_WG + (i * vel) for i in range(1, num_waypoints + 1)]
        self.waypoints_q = [qrob] + [panda.ik(RigidTransform(R_WG, wp)) for wp in pts]

    def __call__(self, state, time):
        if self.start_time is None:
            self.start_time = time
        curr_wp = int((time - self.start_time) / self.time_per_waypoint) + 1
        if curr_wp == len(self.waypoints_q):
            self.done = True
            return PositionAction(state["panda"][:7], self.grasp)
        time_in_wp = time - (self.time_per_waypoint * (curr_wp - 1)) - self.start_time
        assert time_in_wp >= 0
        vel = self.waypoints_q[curr_wp] - self.waypoints_q[curr_wp - 1]
        vel *= self.time_per_waypoint
        qdes = self.waypoints_q[curr_wp - 1] + vel * time_in_wp
        return PositionAction(qdes, self.grasp)
