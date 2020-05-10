# LINKS
# https://towardsdatascience.com/neural-fictitious-self-play-800612b4a53f


from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import sys

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Game name")
flags.DEFINE_integer("num_players", 2, "Number of players")

flags.DEFINE_integer("num_train_episodes", int(1e5),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 100,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")

# DEFINITION: 
# Reinforcement learning algorithms use replay buffers to store trajectories of experience when executing a policy in an environment. 
# During training, replay buffers are queried for a subset of the trajectories (either a sequential subset or a sample) to "replay" the agent's experience.
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")


flags.DEFINE_integer("reservoir_buffer_capacity", int(2e5),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_float("epsilon_start", 0.1, "")
flags.DEFINE_float("epsilon_end", 0.0, "")


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


def runNFSP(hidden_layers_sizes, replay_buffer_capacity, reservoir_buffer_capacity, epsilon_start, epsilon_end, anticipatory_param):
    # Define data storage arrays
    episodes = []
    exploits = []
    nashes = []

    # Initialize the game
    game = FLAGS.game
    num_players = FLAGS.num_players

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.num_train_episodes,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
    }

    # Start the TensorFlow session
    with tf.Session() as sess:
        # Initialize NFSP Agent
        agents = [
            nfsp.NFSP(
                sess, 
                idx, 
                info_state_size, 
                num_actions, 
                hidden_layers_sizes,
                reservoir_buffer_capacity, 
                anticipatory_param,
                **kwargs) for idx in range(num_players)
        ]
        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

        sess.run(tf.global_variables_initializer())
        for ep in range(FLAGS.num_train_episodes):
            # Evaluate Agents
            if ((ep + 1) % FLAGS.eval_every == 0) & ((ep + 1) >= 100):
                losses = [agent.loss for agent in agents]
                logging.info("Losses: %s", losses)
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                nash = exploitability.nash_conv(env.game, expl_policies_avg)
                logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
                logging.info("[%s] NASH AVG %s", ep + 1, nash)
                logging.info("_____________________________________________")

                episodes.append(ep + 1)
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
       
    return episodes, exploits, nashes


# Replay Buffer
def grid_search_repb(colors):
    i = 0

    for x in [50000, 75000, 100000, 200000, 300000]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
            # replay_buffer_capacity = FLAGS.replay_buffer_capacity,
            replay_buffer_capacity = x, 
            reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
            epsilon_start = FLAGS.epsilon_start, 
            epsilon_end = FLAGS.epsilon_end, 
            anticipatory_param = FLAGS.anticipatory_param
        )

        label = "RB size: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('replay_buffer.png')


# Hidden layers
def grid_search_hl(colors):
    i = 0

    for x in [[8], [16], [32], [64], [128]]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in x],
            replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
            reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
            epsilon_start = FLAGS.epsilon_start, 
            epsilon_end = FLAGS.epsilon_end, 
            anticipatory_param = FLAGS.anticipatory_param
        )

        label = "Layer size: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('hidden_layers.png')

# Reservoir buffer
def grid_search_resb(colors):
    i = 0

    for x in [50000, 75000, 100000, 200000, 300000]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
            replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
            reservoir_buffer_capacity = x, 
            epsilon_start = FLAGS.epsilon_start, 
            epsilon_end = FLAGS.epsilon_end, 
            anticipatory_param = FLAGS.anticipatory_param
        )

        label = "ResB size: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('reservoir_buffer.png')


# Anicipatory Param
def grid_search_ap(colors):
    i = 0

    for x in [0.025, 0.05, 0.1, 0.2, 0.3]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
            replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
            reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
            epsilon_start = FLAGS.epsilon_start, 
            epsilon_end = FLAGS.epsilon_end, 
            anticipatory_param = x
        )

        label = "AP: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('anticipatory_param.png')


# Epsilon start
def grid_search_es(colors):
    i = 0

    for x in [0.2, 0.1, 0.05, 0.01, 0.001]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
            replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
            reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
            epsilon_start = x, 
            epsilon_end = FLAGS.epsilon_end, 
            anticipatory_param = FLAGS.anticipatory_param
        )

        label = "Epsilon S: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('epsilon_start.png')


# Epsilon end
def grid_search_ee(colors):
    i = 0

    for x in [0]:
        episodes, exploits, nashes = runNFSP(
            hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
            replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
            reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
            epsilon_start = FLAGS.epsilon_start, 
            epsilon_end = x, 
            anticipatory_param = FLAGS.anticipatory_param
        )

        label = "Epsilon E: " + str(x)
        plt.plot(episodes, exploits, colors[i], label=label)
        i = i + 1
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    plt.ylim(0.01, 10)
    plt.legend(loc='upper right')

    plt.show()
    plt.savefig('epsilon_end.png')


def main(unused_argv):
    colors = ['-r', '-g', '-b', '-m', 'y']

    plt.figure(1)
    grid_search_repb(colors)
    
    plt.figure(2)
    grid_search_resb(colors)

    plt.figure(3)
    grid_search_hl(colors)
    
    plt.figure(4)
    grid_search_ap(colors)
    
    plt.figure(5)
    grid_search_es(colors)
    
    plt.figure(6)
    grid_search_ee(colors)

    # episodes, exploits = runNFSP(
    #     hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
    #     replay_buffer_capacity = FLAGS.replay_buffer_capacity, 
    #     reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
    #     epsilon_start = 0.1, 
    #     epsilon_end = FLAGS.epsilon_end, 
    #     anticipatory_param = FLAGS.anticipatory_param
    # )

    # plt.plot(episodes, exploits)
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.xlim(FLAGS.eval_every, FLAGS.num_train_episodes)
    # plt.ylim(0.01, 10)

    # plt.show()
    # plt.savefig('test.png')



if __name__ == "__main__":
  app.run(main)