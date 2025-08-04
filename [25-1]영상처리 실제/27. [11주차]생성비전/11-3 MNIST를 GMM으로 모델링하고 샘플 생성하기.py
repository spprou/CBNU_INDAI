import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.mixture import GaussianMixture

(x_train, y_train), (x_test, y_test)=mnist.load_data()
X=x_train[np.isin(y_train,[0])]
X=X.reshape((X.shape[0],28*28))

k=8

gm=GaussianMixture(n_components=k).fit(X)

gen=gm.sample(n_samples=10)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for i in range(k):
    plt.subplot(1,10,i+1)
    plt.imshow(gm.means_[i].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()

plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(gen[0][i].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()

