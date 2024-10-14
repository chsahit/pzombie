from typing import List

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


def make_env(env: Env):
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("assets", "assets/")
    parser.AddModels("assets/workspace.dmd.yaml")
    panda = plant.GetModelInstanceByName("panda")
    asset_indices = dict()
    for asset in env.assets:
        asset_idx = parser.AddModels(asset.path)
        assert len(asset_idx) == 1
        asset_indices[asset.name] = asset_idx[0]
    plant.Finalize()
    plant.SetDefaultPositions(panda, env.panda_pose)
    for asset in env.assets:
        plant.SetDefaultPositions(asset_indices[asset.name], asset.pose)
    controller = builder.AddNamedSystem(
        "controller", CartesianStiffnessController(plant)
    )
    policy = builder.AddNamedSystem("policy", PolicySystem(plant, asset_indices))
    builder.Connect(
        plant.get_state_output_port(), policy.get_input_port_estimated_state()
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
    simulator.set_target_realtime_rate(2.0)
    simulator.AdvanceTo(timeout)
