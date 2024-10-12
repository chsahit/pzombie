import numpy as np
from pydrake.all import (
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    Simulator,
    StartMeshcat,
)

from cartesian_stiffness import CartesianStiffnessController
from policy_system import PolicySystem

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


class Env:
    def __init__(self, eraser_pose, panda_pose: np.ndarray = DEFAULT_PANDA_ANGLES):
        self.panda_pose = panda_pose
        self.eraser_pose = eraser_pose


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


def make_env(env: Env):
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("assets", "assets/")
    parser.AddModels("assets/workspace.dmd.yaml")
    panda = plant.GetModelInstanceByName("panda")
    plant.Finalize()
    eraser_idx = plant.GetModelInstanceByName("eraser")
    plant.SetDefaultPositions(panda, env.panda_pose)
    plant.SetDefaultPositions(eraser_idx, env.eraser_pose)
    controller = builder.AddNamedSystem(
        "controller", CartesianStiffnessController(plant)
    )
    num_objects = 1
    num_states = 6 * num_objects + 14
    policy = builder.AddNamedSystem("policy", PolicySystem(plant, 18))
    builder.Connect(
        plant.get_state_output_port(panda), policy.get_input_port_estimated_state()
    )
    builder.Connect(policy.get_output_port(), controller.get_input_port_desired_state())
    builder.Connect(
        plant.get_state_output_port(panda), controller.get_input_port_estimated_state()
    )
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port(panda))
    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    return diagram


def simulate_policy(pi, env, timeout: float = 10.0):
    policy_sys = env.GetSubsystemByName("policy")
    policy_sys.policy = pi
    simulator = Simulator(env)
    simulator.Initialize()
    breakpoint()
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(timeout)
