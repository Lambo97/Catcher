import sys
import numpy as np
import pandas as pd
from random import random, randint
from math import floor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from catcher import ContinuousCatcher

import sys
import csv
import time
import os
import datetime

from collections import deque


class ExtraTreesAgent():
    def __init__(self, env):
        self.render = True
        self.discount_factor = .95
        self.buffer = []
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.model = None
        self.bar_speed = env.bar_speed
        self.env = env
        self.discretization = 7
        self.action_space = np.linspace(-self.bar_speed,
                                        self.bar_speed, self.discretization)

    def memorize(self, state, action, reward, next_state, done):
        # Fill the buffer
        self.buffer[0].append(state)
        self.buffer[1].append(action)
        self.buffer[2].append(reward)
        self.buffer[3].append(next_state)
        self.buffer[4].append(done)

    def train(self):
        csvfile, writer = setup_writer('d_Q', 'extra_trees')
        X = np.hstack((self.buffer[0], self.buffer[1]))
        y = self.buffer[2]
        print("Training prec...")
        self.model = ExtraTreesRegressor().fit(X, y)

        self.model
        dQ = []
        dump(self.model, 'extra_trees/models/Q0.pkl')
        for iteration in range(60):
            y = []
            
            Q_prev = np.zeros((len(self.buffer[0]), len(self.action_space)))
            
            for i, a in enumerate(self.action_space):
                
                testing = np.c_[self.buffer[3], np.repeat(a, len(self.buffer[3]))]
                Q_prev[:, i] = self.model.predict(testing)
            for k, done in enumerate(self.buffer[4]):
                if done:
                    y.append(self.buffer[2][k])
                else:
                    y.append(self.buffer[2][k] + self.discount_factor * np.max(Q_prev[k, :]))
            
            print("Training {}...".format(iteration))
            self.model.fit(X, y)
            dump(self.model, 'extra_trees/models/Q{}.pkl'.format(iteration+1))

            old_model = load('extra_trees/models/Q{}.pkl'.format(iteration))
            d = mean_squared_error(self.model.predict(X), old_model.predict(X))
            print(d)
            writer.writerow([iteration, d])
            self.test(iteration)

    def get_action(self, state):
        if self.model is None:
            action = self.action_space[randint(0, self.discretization - 1)]
            return [action]
        else:
            Q = []
            for a in self.action_space:
                Q.append(self.model.predict([np.hstack((state, [a]))]))
            return [self.action_space[np.argmax(Q)]]
    
    def test(self, iteration):
        fileid = "%s-%d" % ('extra_trees', iteration)
        csvfile, writer = setup_writer(fileid, 'extra_trees')
        for e in range(10):
            done = False
            score = 0
            state = self.env.reset()

            while not done and score < 30000:

                action = agent.get_action(state)
                next_state, reward, done = self.env.step(action)

                score += reward
                state = next_state

                if done or score >= 30000:
                    # every episode, plot the play time
                    scores.append(score)
                    episodes.append(e)
                    writer.writerow([e, score])

        csvfile.close()


def setup_writer(fileid, postfix):
    # we dump episode num, step, total reward, and
    # number of episodes solved in a csv file for analysis
    csvfilename = "%s.csv" % fileid
    csvfilename = os.path.join(postfix, csvfilename)
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile,
                        delimiter=',',
                        quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode',
                     'Total Reward'])

    return csvfile, writer


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    # env = gym.make('Pendulum-v0')
    env = ContinuousCatcher(width=64, height=64)
    # get size of state and action from environment
    state_size = 4
    action_size = 1

    # make A2C agent
    agent = ExtraTreesAgent(env)
    scores, episodes = [], []
    print("Exploring")
    
    for e in range(100):
        done = False
        state = env.reset()
        while not done:

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.memorize(state, action, reward, next_state, done)

            state = next_state
    
    print("Training")
    agent.train()
    print("Testing")
    agent.test(iteration=60)

    csvfile.close()

