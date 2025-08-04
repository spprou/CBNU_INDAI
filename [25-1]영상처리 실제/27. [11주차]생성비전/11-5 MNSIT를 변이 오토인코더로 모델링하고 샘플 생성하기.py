import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

zdim = 32  # latent dimension

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], zdim), mean=0.0, stddev=0.1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Encoder
encoder_input = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(encoder_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Flatten()(x)
z_mean = Dense(zdim)(x)
z_log_var = Dense(zdim)(x)
encoder_output = Lambda(sampling)([z_mean, z_log_var])
model_encoder = Model(encoder_input, [z_mean, z_log_var, encoder_output])

# Decoder
decoder_input = Input(shape=(zdim,))
x = Dense(3136)(decoder_input)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))(x)
decoder_output = x
model_decoder = Model(decoder_input, decoder_output)

# VAE Model
model_input = encoder_input
model_output = model_decoder(encoder_output)
model = Model(model_input, model_output)

reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(model_input, model_output), axis=(1, 2)))
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
total_loss = reconstruction_loss + kl_loss
model.add_loss(total_loss)
model.compile(optimizer='adam')
model.fit(x_train, x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test))

i = np.random.randint(x_test.shape[0])
j = np.random.randint(x_test.shape[0])
x = np.array((x_test[i], x_test[j]))
z = model_encoder.predict(x)[2]

zz = np.zeros((11, zdim))
alpha = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in range(11):
    zz[i] = (1.0-alpha[i]) * z[0] + alpha[i] * z[1]

gen = model_decoder.predict(zz)

plt.figure(figsize=(20, 4))
for i in range(11):
    plt.subplot(1, 11, i+1)
    plt.imshow(gen[i].reshape(28, 28), cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.title(str(alpha[i]))
plt.show()
