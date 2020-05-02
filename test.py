from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_gradient

from metrics import PolicyGradientPolicies
from metrics import plot_exploitability
from metrics import plot_nash_conv
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent.")
flags.DEFINE_integer("num_train_episodes", int(1e5),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def main(_):
  game = "kuhn_poker"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # Plotting
  episodes = []
  exploits = []
  nashs = []

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    agents = [
        dqn.DQN(
            session=sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]

    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    expl_policies_avg = PolicyGradientPolicies(env, agents)

    print("TEST", (agents[0].q_values))

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
        saver.save(sess, FLAGS.checkpoint_dir, ep)

        # Shitty metric code
        losses = [agent.loss for agent in agents]

        # tabular_policy = policy.TabularPolicy(game)
        # print("POL", tabular_policy)

        expl = exploitability.exploitability(env.game, expl_policies_avg)
        nash = exploitability.nash_conv(env.game, expl_policies_avg)
        msg = "EP {}: EXPL: {}, NASH: {}, LOSS: {}\n".format(ep + 1, expl, nash, losses)
        logging.info("%s", msg)

        episodes.append(ep + 1)
        exploits.append(expl)
        nashs.append(nash)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)
  
  plot_exploitability(episodes, exploits)  
  plot_nash_conv(episodes, nashs)


if __name__ == "__main__":
  app.run(main)