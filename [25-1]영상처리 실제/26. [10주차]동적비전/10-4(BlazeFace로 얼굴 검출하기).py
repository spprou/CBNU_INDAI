import cv2 as cv
import mediapipe as mp

img = cv.imread('qwer.jpg')  # 이미지 불러오기

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 얼굴 검출 모델 설정 (model_selection=1: 정밀 모델, confidence=0.5)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 얼굴 검출 수행 (OpenCV는 BGR이므로 RGB로 변환 필요)
res = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# 검출 결과 확인 및 표시
if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img, detection)

    cv.imshow('Face detection by MediaPipe', img)

cv.waitKey()
cv.destroyAllWindows()
