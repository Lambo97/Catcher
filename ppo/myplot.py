import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import glob


cumulated_reward = np.zeros((4, 200, 10))
entropy = np.zeros((4, 200, 10))
loss = np.zeros((4, 200, 10))


for k, filename in enumerate(glob.glob('results/ppo_continuous__*.csv')):
    data = pd.read_csv(filename)
    for epoch in range(200):
        copy = data[data['Epoch'] == epoch]
        cumulated_reward[0, epoch, k] = copy['Cumulated_Reward'].mean()
        entropy[0, epoch, k] = copy['Entropy'].mean()
        loss[0, epoch, k] = copy['Critic Loss'].mean()


for i, n_actions in enumerate([3, 7, 15]):
    j = i+1
    for k, filename in enumerate(glob.glob('results/ppo_discrete_{}_*.csv'.format(n_actions))):
        data = pd.read_csv(filename)
        for epoch in range(200):
            copy = data[data['Epoch'] == epoch]
            cumulated_reward[j, epoch, k] = copy['Cumulated_Reward'].mean()
            entropy[j, epoch, k] = copy['Entropy'].mean()
            loss[j, epoch, k] = copy['Critic Loss'].mean()

plt.plot(range(200), np.mean(cumulated_reward[0], axis=1))
plt.plot(range(200), np.mean(cumulated_reward[1], axis=1))
plt.plot(range(200), np.mean(cumulated_reward[2], axis=1))
plt.plot(range(200), np.mean(cumulated_reward[3], axis=1))
plt.legend(['Continuous', '3 actions', '7 actions', '15 actions'])
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cumulated reward', fontsize=12)
plt.savefig('results/images/cumulated_reward.eps')
plt.show()
plt.close()

plt.plot(range(200), np.mean(entropy[0], axis=1))
plt.plot(range(200), np.mean(entropy[1], axis=1))
plt.plot(range(200), np.mean(entropy[2], axis=1))
plt.plot(range(200), np.mean(entropy[3], axis=1))
plt.legend(['Continuous', '3 actions', '7 actions', '15 actions'])
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.savefig('results/images/entropy.eps')
plt.show()
plt.close()

plt.plot(range(200), np.mean(loss[0], axis=1))
plt.plot(range(200), np.mean(loss[1], axis=1))
plt.plot(range(200), np.mean(loss[2], axis=1))
plt.plot(range(200), np.mean(loss[3], axis=1))
plt.legend(['Continuous', '3 actions', '7 actions', '15 actions'])
plt.ylim([0, 20])
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss of critic network', fontsize=12)
plt.savefig('results/images/loss.eps')
plt.show()
plt.close()