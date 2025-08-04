import os
import xml.etree.ElementTree as ET
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# === 사용자 경로 설정 ===
ANNOTATIONS_DIR = "F:/gjlee/대학원/영상처리 실제/annotations"
IMAGES_DIR = "F:/gjlee/대학원/영상처리 실제/images"
MODEL_SAVE_PATH = "F:/gjlee/대학원/영상처리 실제/2024254020.h5"
LABELS_TXT_PATH = "F:/gjlee/대학원/영상처리 실제/dog_species_name.txt"

# === 어노테이션 파싱 ===
def parse_annotation(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        filename = root.find("filename").text
        if not filename.endswith(".jpg"):
            filename += ".jpg"

        label = root.find("object").find("name").text
        return filename, label
    except:
        return None, None

# === 이미지 전체 경로 매핑 ===
image_map = {}
for dirpath, _, files in os.walk(IMAGES_DIR):
    for file in files:
        if file.endswith(".jpg"):
            image_map[file] = os.path.join(dirpath, file)

# === 어노테이션 매칭 ===
pairs = []
for dirpath, _, files in os.walk(ANNOTATIONS_DIR):
    for file in files:
        xml_path = os.path.join(dirpath, file)
        fname, label = parse_annotation(xml_path)
        if fname in image_map:
            pairs.append((image_map[fname], label))
        else:
            print(f"[❗ 누락 이미지] {fname} → SKIP")

if not pairs:
    raise RuntimeError("❗ 학습 가능한 이미지가 없습니다.")

# === 라벨 인코딩 ===
filepaths, labels = zip(*pairs)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# === TF Dataset 구성 ===
def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image / 255.0, label

dataset = tf.data.Dataset.from_tensor_slices((list(filepaths), list(encoded_labels)))
dataset = dataset.map(load_image).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# === 모델 구성 (Functional API) ===
inputs = tf.keras.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                include_top=False,
                                                weights='imagenet')
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 학습 ===
print(f"🚀 학습 시작 (총 이미지: {len(pairs)})")
model.fit(dataset, epochs=5)

# === 저장 ===
model.save(MODEL_SAVE_PATH)
print(f"✅ 모델 저장 완료: {MODEL_SAVE_PATH}")


# === 견종 이름 저장 코드 여기 추가 ===
with open(LABELS_TXT_PATH, "w", encoding="utf-8") as f:
    for breed in label_encoder.classes_:
        f.write(f"{breed}\n")
print(f"📄 견종 이름 저장 완료: {LABELS_TXT_PATH}")
