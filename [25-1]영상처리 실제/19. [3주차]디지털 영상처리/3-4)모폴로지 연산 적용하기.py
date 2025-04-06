import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 읽기 (Alpha 채널 유지)
img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

# 2. 알파 채널로 이진화 (Otsu's threshold)
t, bin_img = cv.threshold(img[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 3. 이미지의 하단 좌측 절반 부분 잘라내기
b = bin_img[bin_img.shape[0]//2 : bin_img.shape[0], 0 : bin_img.shape[0]//2 + 1]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 4. 구조 요소
se = np.uint8([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])  # 구조 요소 (5x5)

# 5. 팽창
b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 6. 침식
b_erosion = cv.erode(b, se, iterations=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 7. 닫기 연산 (팽창 후 침식)
b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
