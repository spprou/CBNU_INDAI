import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

zdim = 32  # 잠복 공간 차원

# 인코더
encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(encoder_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Flatten()(x)
encoder_output = Dense(zdim)(x)
model_encoder = Model(encoder_input, encoder_output)

# 디코더
decoder_input = Input(shape=(zdim,))
x = Dense(3136)(decoder_input)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
decoder_output = x
model_decoder = Model(decoder_input, decoder_output)

# 오토인코더
model_input = encoder_input
model_output = model_decoder(encoder_output)
model = Model(model_input, model_output)

model.compile(optimizer='Adam', loss='mse')
model.fit(
    x_train, x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)

import matplotlib.pyplot as plt
i = np.random.randint(x_test.shape[0])
j = np.random.randint(x_test.shape[0])
x = np.array((x_test[i], x_test[j]))
z = model_encoder.predict(x)

zz = np.zeros((11, zdim))
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(11):
    zz[i] = (1.0 - alpha[i]) * z[0] + alpha[i] * z[1]

gen = model_decoder.predict(zz)

plt.figure(figsize=(20, 4))
for i in range(11):
    plt.subplot(1, 11, i+1)
    plt.imshow(gen[i].reshape(28, 28), cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(alpha[i]))
