import cv2 as cv
import sys
img=cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일 못찾겠음')

BrushSiz=5 #붓 크기
LColor,RColor=(255,0,0),(0,0,255) #파란색과 빨간색

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1) #왼쪽 클릭시 파란색 원
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,RColor,-1) #오른쪽 클릭시 빨간색 원
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1) #왼쪽 클릭 드래그시 파란색 원
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img,(x,y),BrushSiz,RColor,-1) #오른쪽 클릭 드래그시 빨간색 원

    cv.imshow('Painting',img) #수정된 영상을 다시 그림

cv.namedWindow('Painting')
cv.imshow('Painting',img)

cv.setMouseCallback('Painting',painting)

while True:
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break