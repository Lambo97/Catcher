import sys
import numpy as np
import pandas as pd
from random import random, randint
from math import floor
from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump, load

import sys
import csv
import time
import os
import datetime

from catcher import ContinuousCatcher
from collections import deque


class ExtraTreesAgent():
    def __init__(self, bar_speed=10):
        self.render = False
        self.discount_factor = .95
        self.buffer = deque(maxlen=50000)
        self.model = None
        self.bar_speed = bar_speed
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.discretization = 20
        self.action_space = np.linspace(-bar_speed,
                                        bar_speed, self.discretization)
        self.buffer_size = 10000

    def memorize(self, state, action, reward, next_state):
        # Fill the buffer
        self.buffer.append([state, action, reward, next_state])

    def train(self, iteration):
        # Q1
        if self.model is None:
            X = np.zeros((1, 5))
            y = []
            for i in range(self.buffer_size):
                j = randint(0, len(self.buffer) - 1)
                X = np.vstack(
                    (X, np.hstack((self.buffer[j][0][0], self.buffer[j][1]))))
                y.append(self.buffer[j][2])
            X = X[1:]
            print("Training1...")
            self.model = ExtraTreesRegressor(n_estimators=50, max_depth=20).fit(X, y)
            dump(self.model, 'extra_trees/Q1.pkl')
            return

        x = []
        y = []

        for k in range(self.buffer_size):
            j = randint(0, len(self.buffer) - 1)
            Q_prev = []
            for a in self.action_space:
                Q_prev.append(self.model.predict(
                    [np.hstack((self.buffer[j][3][0], [a]))]))
            y.append(self.buffer[j][2] + self.discount_factor * np.max(Q_prev))
            x.append(np.hstack((self.buffer[j][0][0], self.buffer[j][1])))
        print("Training{}...".format(iteration))
        self.model.fit(x, y)
        self.epsilon *= self.epsilon_decay
        dump(self.model, 'extra_trees/Q{}.pkl'.format(iteration))

    def get_action(self, state):
        if self.model is None or random() < self.epsilon:
            action = self.action_space[randint(0, self.discretization - 1)]
            return [action]
        else:
            Q = []
            for a in self.action_space:
                Q.append(self.model.predict([np.hstack((state[0], [a]))]))
                return [self.action_space[np.argmax(Q)]]


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
                     'Step',
                     'Total Reward',
                     'Number of catch'])

    return csvfile, writer

if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    # env = gym.make('Pendulum-v0')
    env = ContinuousCatcher(width=500, height=500)
    # get size of state and action from environment
    state_size = 4
    action_size = 1

    # make A2C agent
    agent = ExtraTreesAgent(bar_speed=floor(env.bar_speed))

    fileid = "%s-%d" % ('extra_trees', int(time.time()))
    csvfile, writer = setup_writer(fileid, 'extra_trees')
    scores, episodes = [], []

    for e in range(3000):
        done = False
        score = 0
        catch = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        step = 0

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.memorize(state, action, reward, next_state)

            score += reward
            state = next_state
            step += 1
            if reward == 3:
                catch += 1
            if done:
                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                writer.writerow([e, step, score, catch])
                if e % 30 == 0 and e > 1:
                    agent.train(e)
    csvfile.close()
