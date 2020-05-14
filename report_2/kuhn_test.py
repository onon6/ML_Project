import six
import pyspiel

import logging
import sys
from absl import app
from absl import flags
import numpy as np
from six.moves import input
from six.moves import range

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

from utils import play_game
from tournament import tabular_policy_from_csv

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "interactive_play",
    False,
    "Whether to run an interactive play with the agent after training.",
)
flags.DEFINE_integer("games_to_play", 1000, "Number of games to play")


def main(_):
    game = pyspiel.load_game("kuhn_poker")
    action_string = None

    env_configs = {"players": 2}
    env = rl_environment.Environment(game, **env_configs)
    num_actions = env.action_spec()["num_actions"]

    agents = [
        tabular_policy_from_csv(game, "./kuhn_policy.csv"),
    ]

    # Loop through a certain amount of games and play with the trained agent against
    # a command line player or a random player

    # Define end player
    end_player = 1

    # 0: Agent, 1: Other, (2: Ties)
    gains = [0, 0]
    wins = [0, 0, 0]

    for i in range(FLAGS.games_to_play):
        state = game.new_initial_state()
        time_step = env.reset()

        results = play_game(
            game, state, agents, end_player, print_output=False, interactive=False
        )

        gains[0] += results[1 - end_player]
        gains[1] += results[end_player]

        if results[1 - end_player] > results[end_player]:
            wins[0] += 1
        elif results[1 - end_player] == results[end_player]:
            wins[2] += 1
        else:
            wins[1] += 1

        # Switch end player (= switch players P0 becoms P1 and visa versa)
        end_player = 1 - end_player

    for pid in range(game.num_players()):
        print("Final utility for Player {} is {}".format(pid, gains[pid]))

    print("\n")

    for pid in range(game.num_players()):
        print("Player {} won {} out of {} games".format(pid, wins[pid], num_games))

    print("With {} ties out of {} games".format(wins[2], num_games))


if __name__ == "__main__":
    app.run(main)
