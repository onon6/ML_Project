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

FLAGS = flags.FLAGS

flags.DEFINE_boolean("should_train", False, "Game should be trained")
flags.DEFINE_boolean("should_play", False, "Game should be played")

flags.DEFINE_integer("num_episodes", int(1e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play", True,
    "Whether to run an interactive play with the agent after training.")


def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(
                    time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


def main(_):
    game = "kuhn_poker"
    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    print("ENV", dir(env))
    print("TEST", env.get_state)

    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # 1. Train the agents
    if FLAGS.should_train:
        training_episodes = FLAGS.num_episodes
        for cur_episode in range(training_episodes):
            if cur_episode % int(1e4) == 0:
                win_rates = eval_against_random_bots(
                    env, agents, random_agents, 1000)
                logging.info("Starting episode %s, win_rates %s",
                             cur_episode, win_rates)
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

        if not FLAGS.iteractive_play:
            return

    # 2. Play from the command line against the trained agent.
    if FLAGS.should_play:
        human_player = 1
        while True:
            logging.info("You are playing as %s", "1" if human_player else "2")
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == human_player:
                    agent_out = agents[human_player].step(
                        time_step, is_evaluation=True)

                    action = command_line_action(time_step)
                else:
                    agent_out = agents[1 -
                                       human_player].step(time_step, is_evaluation=True)
                    action = agent_out.action
                time_step = env.step([action])

            logging.info("Rewards: Human %s | Agent %s",
                         time_step.rewards[human_player], time_step.rewards[1 - human_player])
            logging.info("End of game!")

            # Switch order of players
            human_player = 1 - human_player


if __name__ == "__main__":
    app.run(main)
