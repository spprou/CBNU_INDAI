import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()
X=x_train[np.isin(y_train,[0])]
X=X.reshape((X.shape[0],28*28))

m=np.mean(X,axis=0)
cv=np.cov(X,rowvar=False)

gen=np.random.multivariate_normal(m,cv,5)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(gen[i].reshape(28,28), cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()