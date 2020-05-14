import six
import pyspiel
import logging
import sys
import numpy as np

from absl import app
from six.moves import input
from six.moves import range

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from tournament import tabular_policy_from_csv


def command_line_action(state, legal_actions):
    # Get a valid action from the command line
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
    # Pretty print the legal action for the sepcific round
    result = [[x, state.action_to_string(state.current_player(), x)] for x in actions]

    output = "\nChoose an action from {}:".format(actions)
    for action in result:
        output = output + "\n" + str(action[0]) + " : " + str(action[1])

    logging.info(output)


def pretty_print_kuhn_state(state):
    # Pretty print the state, more specific the cards dealt to the players
    def num_to_card(str_num):
        # Map the numbers to the cards
        cards = {0: "J", 1: "Q", 2: "K"}
        num = int(str_num)
        argument = int(num)
        return cards.get(argument, "[No card]")

    arr = str(state).split("\n")
    for a in arr:
        if "Cards" in a:
            result = list(map(num_to_card, a.split()[5:8]))
            logging.info(
                "\nPlayer  |  Card \n0       |  %s \n1       |  %s\n\n",
                result[0],
                result[1],
            )


def pretty_print_leduc_state(state):
    # Pretty print the state, more specific the cards dealt to the players
    def num_to_card(str_num):
        # Map the numbers to the cards
        num = int(str_num)
        cards = {0: "JS", 1: "JH", 2: "QS", 3: "QH", 4: "KS", 5: "KH"}
        argument = int(num)
        return cards.get(argument, "[No card]")

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


def play_game(game, state, agents, end_player, print_output=True, interactive=False):
    # Play the selected game (Kuhn or Leduc) and print the state in each round
    def sample_action(state, player_id, end_player):
        # Sample the actions based on the current player
        cur_legal_actions = state.legal_actions(player_id)

        if player_id == end_player:
            if interactive:
                # Interactive play - choose action from command line
                return command_line_action(state, cur_legal_actions)
            else:
                # Random agent - random action from legal actions
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
        if "kuhn" in str(state.get_game()):
            pretty_print_kuhn_state(state)
        else:
            pretty_print_leduc_state(state)

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
            if "kuhn" in str(state.get_game()):
                pretty_print_kuhn_state(state)
            else:
                pretty_print_leduc_state(state)

    # Game is now done. Print utilities for each player
    if not print_output:
        if "kuhn" in str(state.get_game()):
            pretty_print_kuhn_state(state)
        else:
            pretty_print_leduc_state(state)

    returns = state.returns()
    print("Game Ended! (Agent = Player {})".format(1 - end_player))
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))
    # Switch end player
    print("\n===================================\n")
    return returns
