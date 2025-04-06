import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('mistyroad.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #명암 영상으로 변환하고 출력
plt.imshow(gray,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHitst([gray],[0],None,[256],[0,256]) #히스토그램 구하기
plt.plot(h,color='r',linewidth=1),plt.show()

equal=cv.equalizeHist(gray) #히스토그램 평활화하고 출력
plt.imshow(equal,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHist([equal],[0],None,[256],[0,256]) #히스토그램 구하기
plt.plot(h,color='r',linewidth=1),plt.show()
