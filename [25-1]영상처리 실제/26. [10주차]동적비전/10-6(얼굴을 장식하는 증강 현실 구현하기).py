import cv2 as cv
import mediapipe as mp

# 알파 채널 포함한 PNG 이미지 불러오기
dice = cv.imread('qwer simbol.png', cv.IMREAD_UNCHANGED)  # 알파 채널 포함된 PNG
dice = cv.resize(dice, dsize=(0, 0), fx=0.1, fy=0.1)  # 축소
w, h = dice.shape[1], dice.shape[0]

# MediaPipe 설정
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res = face_detection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.detections:
        for det in res.detections:
            # 오른쪽 눈 기준 위치 가져오기
            p = mp_face_detection.get_key_point(det, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            x1, x2 = int(p.x * frame.shape[1] - w / 2), int(p.x * frame.shape[1] + w / 2)
            y1, y2 = int(p.y * frame.shape[0] - h / 2), int(p.y * frame.shape[0] + h / 2)

            # 프레임 내부 좌표인지 확인
            if x1 > 0 and y1 > 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                alpha = dice[:, :, 3:] / 255.0  # 알파채널 추출 (0~1 범위)
                # PNG를 프레임에 알파 블렌딩
                frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha) + dice[:, :, :3] * alpha

    # 좌우 반전 후 출력
    cv.imshow('MediaPipe Face AR', cv.flip(frame, 1))
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
