import numpy as np
import tensorflow as tf
import gym
import time
import csv
from actor_critic import actor_critic, discount_cumsum
from random import random
from sklearn.preprocessing import StandardScaler

from catcher import ContinuousCatcher


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, state_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, state, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if self.ptr >= self.max_size:  # buffer has to have room so you can store
            return
        self.state_buf[self.ptr] = state
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.state_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(env, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=200, gamma=0.95, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=30000,
        target_kl=0.01, render=False, discrete=False, n_actions=7):
    """

    Args:
        env : The environement

        ac_kwargs (dict): Arguments for the actor critic function

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        
        render : Display the environement

    """

    tf.set_random_seed(seed)
    np.random.seed(seed)

    state_dim = 4
    act_dim = 1

    # Share information about action space with policy architecture
    ac_kwargs['bar_speed'] = env.bar_speed

    # Inputs to computation graph
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, state_dim))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
    adv_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
    ret_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
    logp_old_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    # Main outputs from computation graph
    pi, logp, logp_pi, v, entropy = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi, entropy]

    # Experience buffer
    buf = PPOBuffer(state_dim, act_dim, steps_per_epoch, gamma, lam)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    

    # Optimizers
    train_pi = tf.train.AdamOptimizer(pi_lr, name='actor_optimizer').minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(vf_lr, name='critic_optimizer').minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training function
    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = np.mean(kl)
            if kl > 1.5 * target_kl:
                print('Early stopping at step {} due to reaching max kl.'.format(i))
                break
        
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)
        
        r, m_adv, val = sess.run([ratio, min_adv, v], feed_dict=inputs)
        v_loss = np.mean((inputs[ret_ph] - val) ** 2)
        return v_loss


    # Setup statistics
    iteration = 0
    mean_reward = []
    v_loss = 0
    csvfile, writer = setup_writer(discrete, n_actions)

    # Initialize the scaler
    scaler = Scaler(env)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        end_epoch = False
        step_epoch = 0
        # While we do not have collected 4000 transition steps
        while not end_epoch:
            done = False
            # Collect and scale the first observation
            state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            state = scaler.scale_state(state)

            # Set the stats
            discounted = 0
            entropy = []
            iteration += 1

            # Play the environement
            while not done:
                done = False
                if render:
                    env.render()

                action, v_t, logp_t, ent = sess.run(get_action_ops, feed_dict={x_ph: state.reshape(1, -1)})
                # Store what the agent perform
                buf.store(state, action, reward, v_t, logp_t)

                # Perform the action and scale the state
                state, reward, done = env.step(action[0])
                state = scaler.scale_state(state)

                # Update stats
                discounted += (gamma ** ep_len) * reward
                ep_ret += reward
                ep_len += 1
                step_epoch += 1
                entropy.append(ent)
                done = done or (ep_len == max_ep_len)
                if done:
                    mean_reward.append(ep_ret)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = reward if done else sess.run(v, feed_dict={x_ph: state.reshape(1,-1)})
                    buf.finish_path(last_val)
                    ent = np.mean(ent)
                    # Display information
                    print("| iteration: {} | Epoch: {} | Total Reward: {} | Entropy: {:0.2f} | Critic Loss: {:0.5f} | Discounted Reward: {:0.2f}".format(iteration, epoch, ep_ret, ent, v_loss, discounted))
                    writer.writerow([iteration, epoch, ep_ret, ent, v_loss, discounted])
                    if step_epoch >= steps_per_epoch:
                        v_loss = update()
                        saver = tf.train.Saver()
                        #saver.save(sess, "ckpt/model")
                        end_epoch = True


def setup_writer(discrete, n_actions):
    # we dump episode num, step, total reward, and
    # number of episodes solved in a csv file for analysis
    if discrete:
        d = 'discrete'
    csvfilename = "results/ppo_{}_{}_{}.csv".format(d, n_actions, int(time.time()))
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile,
                        delimiter=',',
                        quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Iteration',
                     'Epoch',
                     'Cumulated_Reward',
                     'Entropy',
                     'Critic Loss',
                     'Discounted Reward'])

    return csvfile, writer


class Scaler():
    def __init__(self, env):
        # Normalizer
        # Create a batch of 10000 transitions
        # to fit the normalizer
        state_space_samples = []
        for i in range(10000):
            env.reset()
            done = False
            while not done:
                action = random() * env.bar_speed * 2 - env.bar_speed
                state, reward, done = env.step([action])
                state_space_samples.append(state)

        self.scaler = StandardScaler()
        self.scaler.fit(state_space_samples)

    def scale_state(self, state):
        # requires input shape=(2,)
        scaled = self.scaler.transform([state])
        return scaled


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--actions', type=int, default=7)
    parser.add_argument('--discrete', help='Set discrete action space', action='store_true')
    parser.add_argument('--render', help='Render the environement', action='store_true')
    args = parser.parse_args()

    env = ContinuousCatcher(width=250, height=250)
    tf.reset_default_graph()
    
    ppo(env, gamma=args.gamma, ac_kwargs=dict(discrete=args.discrete, bar_speed=env.bar_speed, action_space=args.actions, ),
        seed=int(time.time()), steps_per_epoch=args.steps, epochs=args.epochs,
        render=args.render, discrete=args.discrete, n_actions=args.actions)

