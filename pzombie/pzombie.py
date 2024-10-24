from pydrake.all import (
    AbstractValue,
    AddDefaultVisualization,
    AddMultibodyPlantSceneGraph,
    ContactResults,
    DiagramBuilder,
    DiscreteContactApproximation,
    Parser,
    Simulator,
    StartMeshcat,
    ZeroOrderHold,
)

from pzombie.cartesian_stiffness import CartesianStiffnessController
from pzombie.components import Env
from pzombie.policy_system import PolicySystem


def _make_env(env: Env):
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
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    # plant.set_contact_model(ContactModel.kPoint)
    plant.Finalize()
    plant.SetDefaultPositions(panda, env.panda_pose)
    for asset in env.assets:
        plant.SetDefaultPositions(asset_indices[asset.name], asset.pose)
    controller = builder.AddNamedSystem(
        "controller", CartesianStiffnessController(plant)
    )
    policy = builder.AddNamedSystem(
        "policy", PolicySystem(plant, asset_indices, scene_graph)
    )
    contact_zoh = builder.AddNamedSystem(
        "zoh", ZeroOrderHold(0.0, AbstractValue.Make(ContactResults()), 0.01)
    )
    builder.Connect(
        plant.get_state_output_port(), policy.get_input_port_estimated_state()
    )
    builder.Connect(
        plant.get_contact_results_output_port(), contact_zoh.get_input_port()
    )
    builder.Connect(contact_zoh.get_output_port(), policy.get_input_port_contact())
    builder.Connect(policy.get_output_port(), controller.get_input_port_desired_state())
    builder.Connect(
        plant.get_state_output_port(panda), controller.get_input_port_estimated_state()
    )
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port(panda))
    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    return diagram


def simulate_policy(pi, env, timeout: float = 10.0, target_rtr: float = 0.0):
    drake_env = _make_env(env)
    policy_sys = drake_env.GetSubsystemByName("policy")
    policy_sys.policy = pi
    simulator = Simulator(drake_env)
    simulator.Initialize()
    simulator.set_target_realtime_rate(target_rtr)
    simulator.AdvanceTo(timeout)
