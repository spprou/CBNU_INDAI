import cv2 as cv
import sys

img=cv.imread('soccer.jpg') # 영상읽기

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('image display', img)  # 윈도우에 영상표시

cv.waitKey()
cv.destroyAllWindows()
