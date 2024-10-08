import numpy as np

import pzombie


def test_simulation():
    def policy(state):
        return np.zeros((20,))

    env = pzombie.make_env()
    pzombie.simulate_policy(policy, env)


if __name__ == "__main__":
    test_simulation()
