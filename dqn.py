# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import pyspiel

import logging
import sys
from absl import app
from absl import flags
import numpy as np
import keras
import random
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from collections import deque

from open_spiel.python import rl_environment

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_spec()["info_state"]
        model.add(Dense(24, input_dim=(state_shape[0] * 2 + 1), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_spec()["num_actions"]))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            num_actions = self.env.action_spec()["num_actions"]
            return np.random.randint(0,num_actions-1)
        info_state_concat = np.array(state.observations["info_state"][0] + state.observations["info_state"][1] + [state.observations["current_player"]])
        return np.argmax(self.model.predict(info_state_concat.reshape(1, -1)))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            info_state_concat = np.array(state.observations["info_state"][0] + state.observations["info_state"][1] + [state.observations["current_player"]]) 
            target = self.target_model.predict(info_state_concat.reshape(1, -1))
            if done:
                target[0][action] = reward[action]
            else:
                info_state_new = np.array(new_state.observations["info_state"][0] + new_state.observations["info_state"][1] + [state.observations["current_player"]])
                Q_future = max(self.target_model.predict(info_state_new.reshape(1, -1)))
                target[0][action] = reward[action] + Q_future[action] * self.gamma
            self.model.fit(info_state_concat.reshape(1, -1), target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

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

def main(_):
    game = "kuhn_poker"
    env = rl_environment.Environment(game)

    logging.info(env)

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    trials = 10000
    trial_len = 500

    for trial in range(trials):
        if trial % 10 == 0:
            print("Trial #{}".format(str(trial)))
        cur_state = env.reset() #reshape

        while True:
            action = dqn_agent.act(cur_state)
            ts = env.step([action])
            new_state = ts
            reward = ts.rewards
            done = ts.last()

            # reward = reward if not done else -20
            new_state = new_state #reshape
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            
            dqn_agent.target_train() # iterates target model
            cur_state = new_state
            if done:
                break

    player_1 = 0
    while True:
        time_step = env.reset()
        pretty_print_state(env)
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == player_1:
                #agent_out = agents[player_1].step(time_step, is_evaluation=True)

                logging.info("Pick action for player %s", player_id)
                action = command_line_action(env, time_step)
            else:
                #agent_out = agents[1 - player_1].step(time_step, is_evaluation=True)

                logging.info("Pick action for player %s", player_id)
                #action = command_line_action(env, time_step)

                action = dqn_agent.act(time_step)

                logging.info("Agent action: %s", action)
            time_step = env.step([action])

        logging.info("Rewards: Player_0 %s | Player_1 %s",
                        time_step.rewards[player_1], time_step.rewards[1 - player_1])
        logging.info("End of game!")


    '''
    game = "kuhn_poker"
    env = rl_environment.Environment(game)

    logging.info(env)

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    trials = 1000
    trial_len = 500

    for trial in range(trials):
        if trial % 10 == 0:
            print("Trial #{}".format(str(trial)))
        cur_state = env.reset() #reshape

        while True:
            action = dqn_agent.act(cur_state)
            ts = env.step([action])
            new_state = ts
            reward = ts.rewards
            done = ts.last()

            # reward = reward if not done else -20
            new_state = new_state #reshape
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            
            dqn_agent.target_train() # iterates target model
            cur_state = new_state
            if done:
                break
            
            
    dqn_agent.save_model('./model.h5')

        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break
    '''

if __name__ == "__main__":
    app.run(main)
