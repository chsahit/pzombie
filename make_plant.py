import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
)


def make_plant(eraser_pose: np.ndarray = np.array([0.1, 0.0, 0.6])):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("assets", "assets/")
    parser.AddModels("assets/workspace.dmd.yaml")
    plant.Finalize()
    eraser_idx = plant.GetModelInstanceByName("eraser")
    plant.SetDefaultPositions(
        eraser_idx,
        np.array([eraser_pose[0], eraser_pose[1], eraser_pose[2], 1, 0, 0, 0]),
    )


if __name__ == "__main__":
    print("testing make_plant")
    make_plant()
