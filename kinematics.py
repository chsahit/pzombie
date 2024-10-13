import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    Parser,
    RigidTransform,
    RotationMatrix,
    Solve,
)


def _make_panda():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.1)
    parser = Parser(plant)
    # parser.package_map().Add("assets", "assets/")
    parser.AddModelsFromUrl(
        "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
    )
    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("panda_link0"),
        X_FM=RigidTransform(),
    )
    plant.Finalize()
    return plant


class Kinematics:
    def __init__(self):
        self.plant = _make_panda()
        self.G = self.plant.GetFrameByName("panda_hand")
        self.B = self.plant.GetFrameByName("panda_link0")
        self.panda = self.plant.GetModelInstanceByName("panda")

    def ik(self, X_BG: RigidTransform) -> np.ndarray:
        ik = InverseKinematics(self.plant)
        ik.AddPositionConstraint(
            self.G,
            [0, 0, 0],
            self.B,
            X_BG.translation(),
            X_BG.translation(),
        )
        ik.AddOrientationConstraint(
            self.G,
            RotationMatrix(),
            self.B,
            X_BG.rotation(),
            0.0,
        )
        prog = ik.get_mutable_prog()
        q = ik.q()
        q0_guess = np.array(
            [
                0.0796904,
                0.18628879,
                -0.07548908,
                -2.42085905,
                0.06961755,
                2.52396334,
                0.6796144,
                0.0,
                0.0,
            ]
        )
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0_guess, q)
        prog.SetInitialGuess(q, q0_guess)
        result = Solve(ik.prog())
        soln = result.GetSolution(q)
        return soln[:7]

    def fk(self, joint_angles: np.ndarray) -> RigidTransform:
        q = np.concatenate([joint_angles, np.array([0, 0])])
        self.plant.SetDefaultPositions(self.panda, q)
        context = self.plant.CreateDefaultContext()
        X_BG = self.plant.CalcRelativeTransform(context, self.B, self.G)
        return X_BG
