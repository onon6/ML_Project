from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import pyspiel
import time
import os

import matplotlib.pyplot as plt
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms import expected_game_score
import pandas as pd
#from tournament import policy_to_csv
import itertools 
import random
from math import floor
tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "leduc_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")

class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def train_network(num_episodes, hidden_layers_sizes, replay_buffer_capacity, reservoir_buffer_capacity, anticipatory_param, epsilon_start):
    logging.info("Training network with hyperparameters: LAY_SIZE={}, REPBUFCAP={}, RESBUFCAP={}, ANTPARAM={}, ESTART={}".format(hidden_layers_sizes, replay_buffer_capacity, reservoir_buffer_capacity, anticipatory_param, epsilon_start))
    game = FLAGS.game_name
    num_players = FLAGS.num_players

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]  

    hidden_layers_sizes = [hidden_layers_sizes]
    kwargs = {
      "replay_buffer_capacity": int(replay_buffer_capacity),
      "reservoir_buffer_capacity": int(reservoir_buffer_capacity),
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": float(anticipatory_param),
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": float(epsilon_start),
      "epsilon_end": FLAGS.epsilon_end,
    } 

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
            **kwargs) for idx in range(num_players)
        ]
        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

        episodes = []
        exploits = []
        nashes = []

        sess.run(tf.global_variables_initializer())

        for ep in range(num_episodes):
            if (ep + 1) % FLAGS.eval_every == 0:
                losses = [agent.loss for agent in agents]
                # logging.info("Losses: %s", losses)
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                nash = exploitability.nash_conv(env.game, expl_policies_avg)
                logging.info("[%s/%s] AVG Exploitability %s", ep + 1, num_episodes, expl)
                
                episodes.append(ep+1)
                exploits.append(expl)
                nashes.append(nash)

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

        policy_to_csv(pyspiel.load_game("leduc_poker"), expl_policies_avg, './best_network_policy.csv')

    return (episodes, exploits, nashes)
    

def random_search():
    HLAY_SIZES = [8,16,32,64,128]
    REPBUFCAP  = [100000, 200000, 300000]
    RESBUFCAP  = [1000000, 2000000, 3000000]
    ANTPARAM   = [0.01, 0.05, 0.1]
    ESTART     = [0.1, 0.06, 0.03]
    allparams  = [HLAY_SIZES] + [REPBUFCAP] + [RESBUFCAP] + [ANTPARAM] + [ESTART]
    len_perms = len(HLAY_SIZES) * len(REPBUFCAP) * len(RESBUFCAP) * len(ANTPARAM) * len(ESTART)
    permutations = list(itertools.product(*allparams)) 
    assert len(permutations) == len_perms # anders hebben we een probleem

    pipeline_idx = floor(len_perms/2)
    danilo_range = range(1,pipeline_idx)
    niels_range = range(pipeline_idx, len_perms)

    #### PAS ENKEL DEZE PARAMETERS AAN
    aantal_evals = 20
    range_idx = niels_range
    num_episodes = int(1e6) #FYI: default was 20e6
    ####

    perm_idxs = random.sample(range_idx, aantal_evals)
    network_summaries = dict()
    ctr = 1
    for perm_idx in perm_idxs:
        logging.info("==================================================")
        logging.info("Starting episode {}/{}".format(ctr, aantal_evals))
        logging.info("==================================================")
        hp = permutations[perm_idx]
        episodes, exploits, nashes = train_network(num_episodes, hp[0], hp[1], hp[2], hp[3], hp[4])
        network_summaries[perm_idx] = exploits[-1]
        filename = './checkpoints/' + str(perm_idx) + '.npy'
        np.save(filename, exploits[-1])
        ctr +=1
    network_summaries["all_perms"] = permutations
    np.save("network_summary.npy", network_summaries)
    

def find_best_network():
    HLAY_SIZES = [8,16,32,64,128]
    REPBUFCAP  = [100000, 200000, 300000]
    RESBUFCAP  = [1000000, 2000000, 3000000]
    ANTPARAM   = [0.01, 0.05, 0.1]
    ESTART     = [0.1, 0.06, 0.03]
    allparams  = [HLAY_SIZES] + [REPBUFCAP] + [RESBUFCAP] + [ANTPARAM] + [ESTART]
    len_perms = len(HLAY_SIZES) * len(REPBUFCAP) * len(RESBUFCAP) * len(ANTPARAM) * len(ESTART)
    permutations = list(itertools.product(*allparams)) 

    directory = os.fsencode('./checkpoints')

    min_expl = float('inf')
    min_file = None
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        expl = np.load('./checkpoints/' + filename)        
        if expl < min_expl:
            min_file = filename
            min_expl = expl
    logging.info("Filename {} contained lowest exploitability of {}".format(min_file, str(min_expl)))   
    splitted = min_file.split('.')
    idx = splitted[0]
    hp_optim = permutations[int(idx)]
    return hp_optim

def save_best_network():
    hp = find_best_network()
    episodes, exploits, nashes = train_network(int(20e6), hp[0], hp[1], hp[2], hp[3], hp[4])
    d = dict()
    d["episodes"] = episodes
    d["exploits"] = exploits
    d["nashes"]   = nashes
    np.save("best_network.npy", d)

def plot_best_network():
    d = np.load('./best_network.npy', allow_pickle=True).item()
    episodes = d["episodes"]
    exploits = d["exploits"]
    nashes   = d["nashes"]
    print(exploits[-1])

    plt.figure()
    plt.plot(episodes, exploits, '-r', label='Exploitability')
    plt.xlabel('Episode')
    plt.ylabel('Exploitability')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('nfsp_exploits.png')

    plt.figure()
    plt.plot(episodes, nashes, '-r', label='NashConv')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('nfsp_nash.png')

def print_average_payouts():
    game = pyspiel.load_game('leduc_poker')
    average_policy = __tabular_policy_from_csv(game, './best_network_policy.csv')
    average_policy_values = expected_game_score.policy_value(game.new_initial_state(), [average_policy] * 2)
    print(average_policy_values)


def __tabular_policy_from_csv(game, filename):
    csv = pd.read_csv(filename, index_col=0)

    empty_tabular_policy = TabularPolicy(game)
    for state_index, state in enumerate(empty_tabular_policy.states):
        action_probabilities = {
                action: probability
                for action, probability in enumerate(csv.loc[state.history_str()])
                if probability > 0
            }
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]
        empty_tabular_policy.action_probability_array[
            state_index, :] = infostate_policy
    return empty_tabular_policy

def main(unused_argv):
    print_average_payouts()

if __name__ == "__main__":
  app.run(main)
