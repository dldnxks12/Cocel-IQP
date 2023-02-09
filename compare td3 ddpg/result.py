import numpy as np
import matplotlib.pyplot as plt


TD3_X = np.load('./TD3_X_mean.npy')
TD3_Y = np.load('./TD3_Y_mean.npy')

DDPG_X = np.load('./DDPG_X_mean.npy')
DDPG_Y = np.load('./DDPG_Y_mean.npy')

plt.xlabel("Episode")
plt.ylabel("Return per episode")
plt.plot(TD3_X, TD3_Y, 'r-', label = 'TD3')
plt.plot(DDPG_X, DDPG_Y, 'b-', label = 'DDPG')
plt.legend(['TD3', 'DDPG'])

plt.show()

# save figure
plt.savefig('Compare')



