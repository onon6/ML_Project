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

from tournament import tabular_policy_from_csv

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "interactive_play",
    False,
    "Whether to run an interactive play with the agent after training.",
)
flags.DEFINE_integer("games_to_play", 100000, "Number of games to play")


def command_line_action(state, legal_actions):
    """Gets a valid action from the user on the command line."""
    action = -1
    while action not in legal_actions:
        pretty_print_actions(state, legal_actions)

        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


def pretty_print_actions(state, actions):
    result = [[x, state.action_to_string(state.current_player(), x)] for x in actions]

    output = "\nChoose an action from {}:".format(actions)
    for action in result:
        output = output + "\n" + str(action[0]) + " : " + str(action[1])

    logging.info(output)


def pretty_print_state(state):
    def num_to_card(str_num):
        num = int(str_num)
        switcher = {0: "JS", 1: "JH", 2: "QS", 3: "QH", 4: "KS", 5: "KH"}

        argument = int(num)
        return switcher.get(argument, "[No card]")

    arr = str(state).split("\n")
    for a in arr:
        if "Cards" in a:
            result = list(map(num_to_card, a.split()[5:8]))
            logging.info(
                "\nPlayer  |  Card \nT       |  %s \n0       |  %s \n1       |  %s\n\n",
                result[0],
                result[1],
                result[2],
            )


def play_game(game, state, agents, end_player, print_output=True):
    def sample_action(state, player_id, end_player):
        cur_legal_actions = state.legal_actions(player_id)

        if player_id == end_player:
            if FLAGS.interactive_play:
                return command_line_action(state, cur_legal_actions)
            else:
                return np.random.choice(cur_legal_actions)
        else:
            # Remove illegal actions, re-normalize probs
            probs = np.zeros(num_actions)
            policy_probs = agents[0].action_probabilities(state, player_id=player_id)
            for action in cur_legal_actions:
                probs[action] = policy_probs[action]
            probs /= sum(probs)
            action = np.random.choice(len(probs), p=probs)
            return action

    # Print the initial state
    if print_output:
      print(str(state))
      pretty_print_state(state)

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            if print_output:
                print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            if print_output:
              print(
                  "Sampled outcome: ",
                  state.action_to_string(state.current_player(), action),
              )
            state.apply_action(action)

        elif state.is_simultaneous_node():
            # Simultaneous node: sample actions for all players.
            chosen_actions = [
                sample_action(state, pid, end_player)
                for pid in xrange(game.num_players())
            ]
            if print_output:
              print(
                  "Chosen actions: ",
                  [
                      state.action_to_string(pid, action)
                      for pid, action in enumerate(chosen_actions)
                  ],
              )
            state.apply_actions(chosen_actions)

        else:
            # Decision node: sample action for the single current player
            action = sample_action(state, state.current_player(), end_player)
            action_string = state.action_to_string(state.current_player(), action)
            if print_output:
              print(
                  "Player ",
                  state.current_player(),
                  ", randomly sampled action: ",
                  action_string,
              )
            state.apply_action(action)

        if print_output:
            print("New state: ", str(state))
            pretty_print_state(state)

    # Game is now done. Print utilities for each player
    if not print_output:
        pretty_print_state(state)
    
    returns = state.returns()
    print("Game Ended! (Agent = Player {})".format(1 - end_player))
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))
    # Switch end player
    print("\n===================================\n")
    return returns


def main(_):
    game = pyspiel.load_game("leduc_poker")
    action_string = None

    env_configs = {"players": 2}
    env = rl_environment.Environment(game, **env_configs)
    num_actions = env.action_spec()["num_actions"]

    agents = [
        tabular_policy_from_csv(game, "./best_network_policy.csv"),
    ]

    # Define end player
    end_player = 1
    num_games = 0

    # 0: Agent, 1: Other, (2: Ties)
    gains = [0, 0]
    wins = [0, 0, 0]

    while num_games < FLAGS.games_to_play:
        state = game.new_initial_state()
        time_step = env.reset()

        results = play_game(game, state, agents, end_player, False)
        
        gains[0] += results[1 - end_player]
        gains[1] += results[end_player]

        if results[1 - end_player] > results[end_player]: 
          wins[0] += 1
        elif results[1 - end_player] == results[end_player]: 
          wins[2] += 1
        else:
          wins[1] += 1

        num_games += 1
        end_player = 1 - end_player

    for pid in range(game.num_players()):
        print("Final utility for Player {} is {}".format(pid, gains[pid]))
    
    print("\n")

    for pid in range(game.num_players()):
        print("Player {} won {} out of {} games".format(pid, wins[pid], num_games))

    print("With {} ties out of {} games".format(wins[2], num_games))


if __name__ == "__main__":
    app.run(main)
