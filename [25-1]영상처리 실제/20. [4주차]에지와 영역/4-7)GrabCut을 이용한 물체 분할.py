import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg') #영상 읽기
img_show=np.copy(img) #붓칠을 디스플레이할 목적의 영상

mask=np.zeros((img.shape[0],img.shape[1]),np.uint8) #마스크 생성
mask[:,:]=cv.GC_PRO_BGD #모든 화소를 배경일 것 같음으로 초기화

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1) #왼쪽 버튼 클릭하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1) #마스크도 그리기
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1) #오른쪽 버튼 클릭하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1) #왼쪽 버튼 클릭하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1) #오른쪽 버튼 클릭하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)

    cv.imshow('Painting',img_show)
    cv.namedWindow('Painting')
    cv.setMoutseCallback('Painting',painting)

    while(True):
        if cv.waitKey(1)==ord('q'):
            break

#여기서부터 Grabcut 적용하는 코드
background=np.zeros((1,65),np.float64) #배경 히스토그램 0으로 초기화
foreground=np.zeros((1,65),np.float64) #물체 히스토그램 0으로 초기화

cv.grabCut(img,mask,None,background,foreground,5,cv.GC_INIT_WITH_MASK) #GrabCut 적용
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD),0,1).astype('uint8')
grab=img*mask2[:,:,np.newaxis]
cv.imshow('Grab cut image',grab)

cv.waitKey()
cv.destroyAllWindows()
