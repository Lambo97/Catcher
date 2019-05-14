import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import glob

window_len = 11


def smooth(x, window_len=window_len, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


cumulated_reward = []
std = []

for i in range(60):
        data = pd.read_csv('results/extra_trees-{}.csv'.format(i))
        cumulated_reward.append(data['Total Reward'].mean())
        std.append(data['Total Reward'].std())

plt.plot(range(len(cumulated_reward)+window_len-1), smooth(cumulated_reward))
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Mean of Cumulated reward', fontsize=12)
plt.savefig('results/mean_cumulated_reward_ET.eps')
plt.show()
plt.close()

plt.plot(range(len(std)+window_len-1), smooth(std))
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Standard deviation of Cumulated reward', fontsize=12)
plt.savefig('results/std_cumulated_reward_ET.eps')
plt.show()
plt.close()

data = pd.read_csv('results/d_Q.csv')
plt.plot(range(len(data['Total Reward'])), data['Total Reward'])
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Standard deviation of Cumulated reward', fontsize=12)
plt.savefig('results/dQ.eps')
plt.show()
plt.close()
