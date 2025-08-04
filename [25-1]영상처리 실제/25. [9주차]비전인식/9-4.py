#  1. 라이브러리 임포트 및 설정
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import os
import random
import cv2 as cv

input_dir = './datasets/oxford_pets/images/images/'
target_dir = './datasets/oxford_pets/annotations/annotations/trimaps/'
img_size = (160, 160)        # 모델 입력 이미지 크기
n_class = 3                  # 클래스 수 (1: 물체, 2: 배경, 3: 경계)
batch_size = 32             # 배치 크기

img_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])
label_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.png') and not f.startswith('.')])


#2. 데이터셋 클래스 정의 (OxfordPets)
class OxfordPets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, img_paths, label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.label_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.img_paths[i : i + self.batch_size]
        batch_label_paths = self.label_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_label_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        y -= 1  # 클래스 레이블 1,2,3 → 0,1,2

        return x, y


def make_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    # U-net의 다운 샘플링 (축소 경로)
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    previous_block_activation = x  # 지름길 연결용

    for filters in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])  # 지름길 연결
        previous_block_activation = x



    #3. U-Net 업샘플링 모델 구성 일부
    # (중간에 Down-sampling 부분은 생략된 상태)
    # Upsampling 경로
    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(n_class, 3, activation='softmax', padding='same')(x)
    model = keras.Model(inputs, outputs)
    return model


#4. 학습 및 예측
model = make_model(img_size, n_class)

# 데이터 셔플 및 훈련/테스트 분리
random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)
test_samples = int(len(img_paths) * 0.1)

train_img_paths = img_paths[:-test_samples]
train_label_paths = label_paths[:-test_samples]
test_img_paths = img_paths[-test_samples:]
test_label_paths = label_paths[-test_samples:]

train_gen = OxfordPets(batch_size, img_size, train_img_paths, train_label_paths)
test_gen = OxfordPets(batch_size, img_size, test_img_paths, test_label_paths)

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 체크포인트 콜백 설정
cb = [keras.callbacks.ModelCheckpoint('oxford_seg.h5', save_best_only=True)]

# 모델 학습
model.fit(train_gen, epochs=30, validation_data=test_gen, callbacks=cb)

# 예측 결과 시각화
preds = model.predict(test_gen)

cv.imshow("Sample image", cv.imread(test_img_paths[0]))
cv.imshow("Segmentation label", cv.imread(test_label_paths[0]) * 64)
cv.imshow("Segmentation prediction", preds[0])
cv.waitKey()
cv.destroyAllWindows()
