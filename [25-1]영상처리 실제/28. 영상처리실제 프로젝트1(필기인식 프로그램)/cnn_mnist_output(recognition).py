import numpy as np
import tensorflow as tf
import cv2 as cv
import os

# 학습된 모델 불러오기
try:
    model = tf.keras.models.load_model('2024254018.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    print("모델 파일('2024254018.h5')이 현재 폴더에 있는지 확인하세요.")
    exit()

# 평가용 이미지 파일 이름
image_filename = 'mnist output test2.png'

# --- 1. 이미지 불러오기 및 전처리 ---
try:
    # 컬러 이미지는 결과 표시에 사용, 그레이스케일 이미지는 숫자 탐지에 사용
    img_color = cv.imread(image_filename)
    if img_color is None:
        raise FileNotFoundError(f"Image file not found at '{image_filename}'")
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
except Exception as e:
    print(f"Error loading image: {e}")
    print(f"이미지 파일('{image_filename}')이 현재 폴더에 있는지 확인하세요.")
    exit()

# 이미지를 흑백으로 변환 (하얀 배경에 검은 글씨 기준)
# cv.threshold의 THRESH_BINARY_INV는 배경(흰색)을 검은색으로, 객체(검은색)를 흰색으로 바꿈
# 모델 학습 시 (MNIST) 흰색 글씨/검은 배경을 사용했기 때문
_, img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)

# --- 2. 이미지에서 숫자 영역 찾기 (Contour Detection) ---
# findContours는 이미지에서 윤곽선(경계)을 찾아줌
# cv.RETR_EXTERNAL: 가장 바깥쪽의 외곽선만 찾음
# cv.CHAIN_APPROX_SIMPLE: 외곽선 꼭짓점만 저장하여 메모리 절약
contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 각 숫자의 경계 상자(bounding box) 정보를 저장할 리스트
bounding_boxes = []
for contour in contours:
    # boundingRect는 해당 외곽선을 감싸는 가장 작은 사각형을 반환 (x, y, 너비, 높이)
    x, y, w, h = cv.boundingRect(contour)
    # 너무 작은 노이즈는 무시 (예: 너비와 높이가 10픽셀 이하)
    if w > 10 and h > 10:
        bounding_boxes.append((x, y, w, h))

# --- ★★★ 핵심 수정 부분: 바운딩 박스를 위->아래, 왼->오른쪽 순으로 정렬 ★★★ ---
if bounding_boxes:
    # 모든 박스들의 평균 높이를 계산하여, 같은 줄에 있는지 판단하는 기준으로 삼음
    avg_height = sum([h for _, _, _, h in bounding_boxes]) / len(bounding_boxes)

    # y좌표를 기준으로 먼저 줄을 나누고, 같은 줄 안에서는 x좌표로 정렬
    # 각 박스의 중심점(y + h/2)을 평균 높이의 0.8배로 나눈 몫으로 줄 번호를 부여
    # 이렇게 하면 세로 위치가 비슷한 박스들이 같은 줄 번호를 갖게 됨
    # 정렬 우선순위: 1. 줄 번호 (오름차순), 2. x좌표 (오름차순)
    bounding_boxes.sort(key=lambda box: (int((box[1] + box[3] / 2) / (avg_height * 0.8)), box[0]))
else:
    print("이미지에서 숫자를 찾을 수 없습니다. 이미지의 글씨가 너무 옅거나 작지 않은지 확인하세요.")
    exit()

# --- 3. 각 숫자 이미지를 모델 입력에 맞게 준비 ---
numerals_for_model = []
for (x, y, w, h) in bounding_boxes:
    # 이진화된 이미지에서 숫자 부분만 잘라내기 (ROI: Region of Interest)
    roi = img_binary[y:y + h, x:x + w]

    # MNIST 크기(28x28)에 맞추기 위해 여백(padding) 추가하여 정사각형 만들기
    # 숫자 왜곡을 방지하기 위함
    old_size = roi.shape[:2]  # (높이, 너비)
    desired_size = 28
    ratio = float(desired_size - 10) / max(old_size)  # 5픽셀 여백 고려
    new_size = tuple([int(dim * ratio) for dim in old_size])

    roi = cv.resize(roi, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 검은색 여백 추가
    padded_roi = cv.copyMakeBorder(roi, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])

    numerals_for_model.append(padded_roi)

if not numerals_for_model:
    print("이미지에서 숫자를 찾을 수 없습니다.")
else:
    # --- 4. 신경망으로 숫자 인식 수행 ---
    numerals_np = np.array(numerals_for_model)
    numerals_np = numerals_np.reshape(len(numerals_np), 28, 28, 1)
    numerals_np = numerals_np.astype(np.float32) / 255.0

    # 모델 예측
    predictions = model.predict(numerals_np)
    class_ids = np.argmax(predictions, axis=1)

    # --- 5. 결과 출력 및 저장 ---
    # 화면에 인식 결과 표시
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        result_text = str(class_ids[i])
        # 원본 컬러 이미지에 인식된 숫자 텍스트를 그림
        cv.putText(img_color, result_text, (x + w // 2 - 10, y + h + 25), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 0, 0), 2)
        # 인식된 영역에 사각형 그리기 (확인용)
        cv.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # txt 파일로 결과 저장
    result_string = "".join(map(str, class_ids))
    with open('2024254018.txt', 'w') as f:
        f.write(result_string)

    print(f"인식된 숫자: {result_string}")
    print("결과가 '2024254018.txt' 파일로 저장되었습니다.")

    # 결과 이미지 보여주기
    cv.imshow('Recognition Result', img_color)
    cv.waitKey(0)  # 아무 키나 누를 때까지 대기
    cv.destroyAllWindows()
