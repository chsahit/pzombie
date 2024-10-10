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


def make_env(eraser_pose: np.ndarray = np.array([0.1, 0.0, 1.6])):
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("assets", "assets/")
    parser.AddModels("assets/workspace.dmd.yaml")
    panda = plant.GetModelInstanceByName("panda")
    # eraser = plant.GetModelInstanceByName("eraser")
    plant.Finalize()
    eraser_idx = plant.GetModelInstanceByName("eraser")
    plant.SetDefaultPositions(
        eraser_idx,
        np.array([eraser_pose[0], eraser_pose[1], eraser_pose[2], 1, 0, 0, 0]),
    )
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
    simulator.AdvanceTo(timeout)
