import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# === 사용자 설정 ===
MODEL_PATH = "F:/gjlee/대학원/영상처리 실제/2024254020.h5"
LABELS_PATH = "F:/gjlee/대학원/영상처리 실제/dog_species_name.txt"
INPUT_IMAGE_PATH = "F:/gjlee/대학원/영상처리 실제/test.jpg"
OUTPUT_IMAGE_PATH = "2024254020.png"

# === 모델 및 라벨 로드 ===
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# === 이미지 전처리 ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

img_tensor = preprocess_image(INPUT_IMAGE_PATH)

# === 예측 ===
preds = model.predict(img_tensor)[0]
top5_indices = np.argsort(preds)[-5:][::-1]
top5_probs = preds[top5_indices]
top5_labels = [class_names[i] for i in top5_indices]

# === 원본 이미지 로드 (표시용)
original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(original_image)

# 폰트 설정 (한글 경로가 문제될 경우 기본폰트 사용)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# === 텍스트 오버레이 ===
text_y = 10
for i in range(5):
    label = top5_labels[i]
    prob = top5_probs[i]
    text = f"{i+1}. {label} ({prob:.2%})"
    draw.text((10, text_y), text, fill=(255, 255, 255), font=font)
    text_y += 25

# === 결과 저장 ===
original_image.save(OUTPUT_IMAGE_PATH)
print(f"✅ 예측 결과 저장 완료: {OUTPUT_IMAGE_PATH}")
