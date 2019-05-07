import numpy as np
import tensorflow as tf
import scipy.signal

EPS = 1e-8


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]




"""
Actor-Critics
"""


def actor_critic(x, a, policy=None, discrete=False, action_space=None, bar_speed=2):

    # default policy builder depends on action space
    if policy is None and not discrete:
        policy = mlp_gaussian_policy
    elif policy is None and discrete:
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi, entropy = policy(x, a, action_space, bar_speed)
    with tf.variable_scope('v'):
        v = tf.squeeze(critic_network(x), axis=1)
    return pi, logp, logp_pi, v, entropy
    
"""
Policies
"""


def mlp_categorical_policy(x, a, action_space, bar_speed=2):
    # We discretize the action space
    action_space = 7
    actions = tf.linspace(-bar_speed, bar_speed, action_space)

    # Compute the probability distribution of the current state
    proba = actor_network_discrete(x, action_space)
    dist = tf.distributions.Categorical(probs=proba)

    # Output the best action
    index = dist.sample(1)
    pi = actions[index[0][0]]
    pi = tf.reshape(pi, [1, 1])

    # Compute the log probability of the action passed as arguments
    w = tf.where(tf.equal(actions, a))
    w = tf.reshape(w[:, 1], [tf.shape(w)[0], 1])
    logp = tf.reduce_sum(tf.multiply(tf.reshape(tf.one_hot(w, depth=action_space), tf.shape(proba)), tf.log(proba)), axis=1)

    # Compute the log probability of the chosen action
    logp_pi = tf.reduce_sum(tf.one_hot(index, depth=action_space) * tf.log(proba), axis=1)
    logp_pi = tf.reduce_sum(logp_pi, axis=1)

    return pi, logp, logp_pi, dist.entropy()


def mlp_gaussian_policy(x, a, action_space, bar_speed=2):
    # Compute the mean and the standard deviation of the probability distribution
    mu, std = actor_network_continu(x)

    # Scale the mean
    mu *= bar_speed
    log_std = tf.log(std + 1e-10)

    # Sample the action
    pi = tf.random_normal(tf.shape(mu), mean=mu, stddev=std)

    # Compute the log probabilities and the entropy of the distribution
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    entropy = tf.distributions.Normal(loc=mu, scale=std).entropy()
    return pi, logp, logp_pi, entropy


def actor_network_discrete(x, action_space=1):
    x = tf.layers.dense(x, units=20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, units=20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, units=action_space, activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def actor_network_continu(x, action_space=1):
    x = tf.layers.dense(x, units=50, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, units=50, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    mu = tf.layers.dense(x, units=action_space, activation='tanh', kernel_initializer=tf.contrib.layers.xavier_initializer())
    sigma = tf.layers.dense(x, units=action_space, activation='softplus', kernel_initializer=tf.contrib.layers.xavier_initializer())
    return mu, sigma


def critic_network(x):
    x = tf.layers.dense(x, units=50, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, units=50, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, units=1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x
