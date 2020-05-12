from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import pyspiel
import pandas as pd

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import matplotlib.pyplot as plt
from tournament import policy_to_csv, tabular_policy_from_csv, play_match
from open_spiel.python.policy import TabularPolicy, tabular_policy_from_policy

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from tournament import policy_to_csv

FLAGS = flags.FLAGS


def pretty_print_state(state):
    cards = str(state).split()[0:2]
    def num_to_card(num):
        switcher = {
            0: "J",
            1: "Q",
            2: "K"
        }

        argument = int(num)
        return switcher.get(argument, "Invalid card")

    result = list(map(num_to_card, cards))
    logging.info(
        "\nPlayer  |  Card \n0       |  %s \n1       |  %s", result[0], result[1])

def main(unused_argv):


    game = pyspiel.load_game("kuhn_poker")
    '''
    cfr_solver = cfr.CFRSolver(game)

    episodes = []
    exploits = []
    nashes = []
    flags.DEFINE_integer("num_train_episodes", int(100000),
                     "Number of training episodes.")
    flags.DEFINE_integer("eval_every", 100,
                     "Episode frequency at which the agents are evaluated.")
    for ep in range(FLAGS.num_train_episodes):
        cfr_solver.evaluate_and_update_policy()
        avg_pol = cfr_solver.average_policy()
        expl = exploitability.exploitability(game, avg_pol)
        nash = exploitability.nash_conv(game, avg_pol)

        exploits.append(expl)
        nashes.append(nash)
        episodes.append(ep)

    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(game.new_initial_state(), [average_policy] * 2)
    
    cur_pol = cfr_solver.current_policy()
    
    plt.plot(episodes, exploits, '-r', label='Exploitability')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    #plt.ylim(0.01, 10)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('cfr_expl.png')

    plt.figure()

    plt.plot(episodes, nashes, '-r', label='NashConv')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('cfr_nash.png')    

    print(average_policy)
    print(average_policy_values)
    policy_to_csv(game, average_policy, './policy_test.csv')

    '''
    policy = tabular_policy_from_csv(game, './policy_test.csv')

    df = pd.DataFrame(
            data=policy.action_probability_array,
            index=[str(s) for s in policy.states])

    print(df)

    #for state in policy.states:
    #    print(str(state))

    #teststate = policy.states[6]
    #print(str(teststate))

if __name__ == "__main__":
    app.run(main)

