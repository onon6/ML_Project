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
from keras.models import Sequential
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
        model.add(Dense(24, input_dim=(state_shape[0] * 2), activation="relu"))
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
        info_state_concat = np.array(state.observations["info_state"][0] + state.observations["info_state"][1])
        print(".................{}".format(info_state_concat.shape))
        return np.argmax(self.model.predict(info_state_concat))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            info_state_concat = np.array(state.observations["info_state"][0] + state.observations["info_state"][1]) 
            target = self.target_model.predict(info_state_concat)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(info_state_concat, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main(_):
    game = "kuhn_poker"
    env = rl_environment.Environment(game)

    logging.info(env)
    time_step = env.reset()
    dqn = DQN(env)

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    trials = 1000
    trial_len = 500

    for trial in range(trials):
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
            
            '''
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break'''


if __name__ == "__main__":
    app.run(main)
