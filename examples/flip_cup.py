import numpy as np

from pzombie import actions, components, pzombie

q0_panda = np.array(
    [
        -0.8931614063267244,
        1.0826762628018491,
        0.5767242533461068,
        -2.3865885852613826,
        2.5576139619935665,
        1.5752390088358155,
        0.18088429190114383,
        0.05,
        0.05,
    ]
)

grasp_pose = np.copy(q0_panda[:7])
q_lift = np.array(
    [
        -0.8476724329427828,
        0.5063150040673973,
        0.7137957234127175,
        -2.6367048674229223,
        2.8973,
        1.671733816523151,
        0.44364635838755134,
    ]
)

q_flip = np.copy(q_lift)
q_flip[-1] += np.pi


def open_loop_flip_policy(state, time):
    if time < 5.0:  # grasp the cup
        return actions.PositionAction(grasp_pose, components.GRIPPER_CLOSE)
    elif time < 13.0:  # lift the gripper
        return actions.PositionAction(q_lift, components.GRIPPER_CLOSE)
    elif time < 18.0:  # flip the hand
        return actions.PositionAction(q_flip, components.GRIPPER_CLOSE)
    else:
        return actions.PositionAction(q_flip, components.GRIPPER_OPEN)


def make_environment():
    cup = components.Asset(
        "cup", "assets/cup.urdf", np.array([1, 0, 0, 0, 0.3, 0.0, 0.71])
    )
    env = components.Env([cup], panda_pose=q0_panda)
    pzombie.simulate_policy(open_loop_flip_policy, env, timeout=40.0, target_rtr=1.0)


def flip_cup():
    make_environment()


if __name__ == "__main__":
    flip_cup()
