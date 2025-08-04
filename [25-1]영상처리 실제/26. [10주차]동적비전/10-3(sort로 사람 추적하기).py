import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names


# YOLO 객체 검출 함수 (res 추출용) — 함수는 이미지에는 없으니 아래에 가정 정의
def yolo_detect(img, model, out_layers):
    height, width = img.shape[:2]
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward(out_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        results.append([x, y, x + w, y + h, confidences[i], class_ids[i]])

    return results

model, out_layers, class_names = construct_yolo_v3()
colors = np.random.uniform(0, 255, size=(100, 3))

from sort import Sort                # sort.py 모듈 불러옴
sort = Sort()                        # Sort 클래스 객체 생성

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 연결
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    res = yolo_detect(frame, model, out_layers)
    persons = [res[i] for i in range(len(res)) if res[i][5] == 0]  # 부류 0은 사람

    if len(persons) == 0:
        tracks = sort.update()  # 사람 검출이 없으면 빈 배열로 업데이트)
    else:
        tracks = sort.update(np.array(persons))

    for i in range(len(tracks)):
        x1, y1, x2, y2, track_id = tracks[i].astype(int)
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[track_id], 2)
        cv.putText(frame, str(track_id), (x1+10, y1+40), cv.FONT_HERSHEY_PLAIN, 3, colors[track_id], 2)

    cv.imshow('Person tracking by SORT', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
