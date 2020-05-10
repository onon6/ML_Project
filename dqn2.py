import pyspiel

import logging
import sys
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn


def main(_):
    env = rl_environment.Environment("kuhn_poker")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]







if __name__ == "__main__":
    app.run(main)
