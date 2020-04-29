from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1000), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play", True,
    "Whether to run an interactive play with the agent after training.")


class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    # self._policies = nfsp_policies
    # self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    # cur_player = state.current_player()
    # legal_actions = state.legal_actions(cur_player)

    # self._obs["current_player"] = cur_player
    # self._obs["info_state"][cur_player] = (
    #     state.information_state_tensor(cur_player))
    # self._obs["legal_actions"][cur_player] = legal_actions

    # info_state = rl_environment.TimeStep(
    #     observations=self._obs, rewards=None, discounts=None, step_type=None)

    # p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    # prob_dict = {action: p[action] for action in legal_actions}
    prob_dict = {0: 100, 1: 200}
    print("PORB", prob_dict)
    return prob_dict


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


def main(_):
    game = "kuhn_poker"
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # Train the agents
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


if __name__ == "__main__":
  app.run(main)