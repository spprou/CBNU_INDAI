# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
from ultralytics import YOLO
import os


def main():
    # --- 1. 로컬 YOLO 모델 로드 ---
    try:
        model_path = 'weights.pt'
        model = YOLO(model_path)
        print(f"'{model_path}' 에서 로컬 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("모델 파일이 프로젝트 폴더 안에 있는지, 파일 이름이 올바른지 확인하세요.")
        return

    # --- 2. 카메라 연결 ---
    DevList = mvsdk.CameraEnumerateDevice()
    if not DevList:
        print("카메라를 찾을 수 없습니다!")
        return

    hCamera = 0
    try:
        DevInfo = DevList[0]
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print(f"카메라 열기 실패({e.error_code}): {e.message}")
        return

    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraPlay(hCamera)

    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    WINDOW_NAME = "Real-time Defect Detection (Offline) | q or Esc to quit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("카메라가 성공적으로 연결되었습니다. 실시간 검출을 시작합니다...")

    # --- ✨ 변경된 부분 시작 ✨ ---
    # 1. 클래스별 색상을 미리 정의합니다. (BGR 순서: 파랑, 초록, 빨강)
    colors = {
        'dent': (255, 0, 0),  # 파란색
        'impurity': (204, 0, 204)  # 보라색
    }
    # --- ✨ 변경된 부분 끝 ✨ ---

    # --- 3. 실시간 검출 루프 ---
    while True:
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            frame = np.frombuffer(
                (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer), dtype=np.uint8
            ).reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if monoCamera else 3))

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            results = model(frame, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # --- ✨ 변경된 부분 시작 ✨ ---
                # 2. 클래스 이름에 맞는 색상을 가져옵니다.
                #    만약 정의되지 않은 클래스일 경우, 기본값으로 빨간색(0,0,255)을 사용합니다.
                color = colors.get(class_name, (0, 0, 255))
                # --- ✨ 변경된 부분 끝 ✨ ---

                # 3. 가져온 색상으로 네모 상자와 텍스트를 그립니다.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {confidence:.2f}"
                font_scale = 3.0
                font_thickness = 3

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, font_thickness)

            annotated_frame = frame

            display_frame = cv2.resize(annotated_frame, (1600, 1200), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, display_frame)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"이미지 획득 실패: {e}")
            continue
        except Exception as e:
            print(f"오류 발생: {e}")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # --- 4. 종료 처리 ---
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")


if __name__ == "__main__":
    main()