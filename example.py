import numpy as np

import pzombie


def test_simulation():
    def policy(state):
        return pzombie.PositionAction(pzombie.DEFAULT_PANDA_ANGLES[:7], 0.0)

    env = pzombie.Env(eraser_pose=np.array([1, 0, 0, 0, 0.6, 0.0, 1.1]))
    drake_env = pzombie.make_env(env)
    pzombie.simulate_policy(policy, drake_env)


if __name__ == "__main__":
    test_simulation()
