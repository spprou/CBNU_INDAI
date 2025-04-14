import cv2 as cv
import numpy as np

# 1. 이미지 로딩 및 전처리
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 모델 이미지: 버스 부분
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv.imread('mot_color83.jpg')  # 장면 이미지
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 검출기 생성 및 특징점 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 3. FLANN 기반 매칭
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 4. Lowe's ratio test 적용
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if nearest1.distance / nearest2.distance < T:
        good_match.append(nearest1)

# 5. 좋은 매칭쌍의 좌표 추출
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])

# 6. RANSAC으로 호모그래피 행렬 추정
H, _ = cv.findHomography(points1, points2, cv.RANSAC)

# 7. 박스 경계선 추정 및 그리기
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
box1 = np.float32([[0,0], [0,h1-1], [w1-1,h1-1], [w1-1,0]]).reshape(4, 1, 2)
box2 = cv.perspectiveTransform(box1, H)
img2 = cv.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)

# 8. 매칭 결과 이미지 생성
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match,
               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 9. 결과 출력
cv.imshow('Matches and Homography', img_match)
cv.waitKey()
cv.destroyAllWindows()
