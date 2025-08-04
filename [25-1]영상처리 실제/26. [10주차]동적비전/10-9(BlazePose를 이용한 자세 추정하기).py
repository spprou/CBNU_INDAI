import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Pose 모델 초기화
pose = mp_pose.Pose(
    static_image_mode=False,             # 영상 스트림용
    enable_segmentation=True,            # 전경/배경 분리
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # RGB 변환 후 포즈 인식 (각 관절의 (x, y, z) 좌표 포함)
    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # 관절 위치 시각화
    mp_drawing.draw_landmarks(
        frame,
        res.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    # 결과 영상 출력
    cv.imshow('MediaPipe pose', cv.flip(frame, 1))  # 좌우 반전

    if cv.waitKey(5) == ord('q'):
        # 종료 시 3D 포즈도 시각화 가능 (plot_landmarks)
        mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        break

cap.release()
cv.destroyAllWindows()
