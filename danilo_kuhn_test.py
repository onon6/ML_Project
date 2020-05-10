from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import matplotlib.pyplot as plt
from tournament import policy_to_csv, tabular_policy_from_csv, play_match

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Game name")
flags.DEFINE_integer("num_players", 2, "Number of players")

flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 100,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    32,
], "Number of hidden units in the avg-net and Q-net.")

# DEFINITION: 
# Reinforcement learning algorithms use replay buffers to store trajectories of experience when executing a policy in an environment. 
# During training, replay buffers are queried for a subset of the trajectories (either a sequential subset or a sample) to "replay" the agent's experience.
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of the replay buffer.")


flags.DEFINE_integer("reservoir_buffer_capacity", int(1e3),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_float("epsilon_start", 0.5, "")
flags.DEFINE_float("epsilon_end", 0.0001, "")

flags.DEFINE_string("modeldir", "./", "directory")


def command_line_action(env, time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]

    action = -1
    while action not in legal_actions:
        pretty_print_actions(env, legal_actions)

        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action

def pretty_print_actions(env, actions):
    result = [[x, env.get_state.action_to_string(x)] for x in actions]

    output = "\nChoose an action from {}:".format(actions)
    for action in result:
        output = output + "\n" + str(action[0]) + " : " + str(action[1])

    logging.info(output)


def pretty_print_state(env):
    state = env.get_state
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
                logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
                logging.info("_____________________________________________")

                episodes.append(ep + 1)
                exploits.append(expl)

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

        for pid, agent in enumerate(agents):
            policy_to_csv(env.game, expl_policies_avg, f"{FLAGS.modeldir}/test_p{pid+1}.csv")
        play(agents, env)
       

    return episodes, exploits



def play(agents, env):
    player_1 = 0
    while True:
        time_step = env.reset()
        pretty_print_state(env)
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == player_1:
                agent_out = agents[player_1].step(
                    time_step, is_evaluation=True)

                logging.info("Pick action for player %s", player_id)
                action = command_line_action(env, time_step)
            else:
                agent_out = agents[1 -
                                    player_1].step(time_step, is_evaluation=True)

                logging.info("Pick action for player %s", player_id)
                #action = command_line_action(env, time_step)

                action = agent_out.action
                logging.info("Agent action: %s", action)
            time_step = env.step([action])

        logging.info("Rewards: Player_0 %s | Player_1 %s",
                        time_step.rewards[player_1], time_step.rewards[1 - player_1])
        logging.info("End of game!")


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
    '''
    episodes, exploits = runNFSP(
        hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes],
        replay_buffer_capacity = FLAGS.replay_buffer_capacity,
        reservoir_buffer_capacity = FLAGS.reservoir_buffer_capacity, 
        epsilon_start = FLAGS.epsilon_start, 
        epsilon_end = FLAGS.epsilon_end, 
        anticipatory_param = FLAGS.anticipatory_param
    )
    '''
    game = pyspiel.load_game(FLAGS.game)
    state = game.new_initial_state()
    info_state_str = state.information_state_string()

    a = tabular_policy_from_csv(game, './test_p1.csv')
    s_policy = a.policy_for_key(info_state_str)

    #play_match(game, './test_p1.csv', './test_p2.csv')


if __name__ == "__main__":
    app.run(main)
