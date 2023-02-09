import numpy as np
import matplotlib.pyplot as plt


X1 = np.load('./TD3_1_X.npy')
X2 = np.load('./TD3_2_X.npy')
X3 = np.load('./TD3_3_X.npy')
X4 = np.load('./TD3_4_X.npy')

Y1 = np.load('./TD3_1_Y.npy')
Y2 = np.load('./TD3_2_Y.npy')
Y3 = np.load('./TD3_3_Y.npy')
Y4 = np.load('./TD3_4_Y.npy')

X = (X1 + X2 + X3 + X4) / 4
Y = (Y1 + Y2 + Y3 + Y4) / 4

np.save('./TD3_X_mean', X)
np.save('./TD3_Y_mean', Y)

plt.plot(X, Y)
plt.show()
plt.savefig('./TD3.png')