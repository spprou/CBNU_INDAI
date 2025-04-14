import cv2 as cv

# 이미지 읽기
img = cv.imread('mot_color70.jpg')  # 해당 경로에 이미지가 있어야 함
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 특징점 검출 및 디스크립터 계산
kp, des = sift.detectAndCompute(gray, None)

# 특징점 시각화 (크기 및 방향 포함)
gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 이미지 출력
cv.imshow('sift', gray)
cv.waitKey()
cv.destroyAllWindows()
