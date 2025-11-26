# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import os
from datetime import datetime


def main_loop():
    # --- 추가된 부분: 이미지 저장 폴더 설정 ---
    save_dir = "captures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"'{save_dir}' 폴더를 생성했습니다.")
    # --- 추가된 부분 끝 ---

    # 카메라 장치 목록 검색
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("카메라를 찾을 수 없습니다!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("사용할 카메라를 선택하세요: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 카메라 열기
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("카메라 열기 실패({}): {}".format(e.error_code, e.message))
        return

    # 카메라 속성(Capability) 가져오기
    cap = mvsdk.CameraGetCapability(hCamera)

    # 흑백 카메라인지 컬러 카메라인지 확인
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 흑백 카메라는 ISP가 MONO 데이터를 바로 출력하도록 설정
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 카메라 모드를 연속 촬영 모드로 변경
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # SDK 내부의 이미지 획득 스레드 시작
    mvsdk.CameraPlay(hCamera)

    # RGB 버퍼 할당
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    # --- 수정된 부분: 키 입력 처리 로직 변경 ---
    while True:
        # 카메라에서 이미지 한 프레임 가져오기
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if monoCamera else 3))

            # 화면 표시용으로 이미지 크기 조절 (원본은 그대로 유지)
            display_frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("s: Save | q: Quit", display_frame)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("이미지 버퍼 획득 실패({}): {}".format(e.error_code, e.message))
            continue  # 에러 발생 시 다음 프레임으로 넘어감

        # 키보드 입력 대기
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break
        # 's' 키를 누르면 이미지 저장
        elif key == ord('s'):
            # 파일 이름에 사용할 현재 시간 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"defect_{timestamp}.jpg")

            # 원본 고해상도 이미지(frame)를 저장
            cv2.imwrite(filename, frame)
            print(f"이미지 저장 완료: {filename}")
    # --- 수정된 부분 끝 ---

    # 카메라 닫기
    mvsdk.CameraUnInit(hCamera)

    # 프레임 버퍼 메모리 해제
    mvsdk.CameraAlignFree(pFrameBuffer)


def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()


main()