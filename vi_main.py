import cv2
import numpy as np
import time
import math
from Drone_Pack import drone_position
import requests
from bs4 import BeautifulSoup

#레이더 임의 거리 값
lider = 50;

#드론 좌표 임의값
drone_x, drone_y, drone_z = 30, 30, 30;
drone_center = (drone_x, drone_y, drone_z);
drone_direction = 10;

# 영상 / 캠 정보받기
#video_file = "go_yolo.mp4"
#cam = cv2.VideoCapture(video_file)
cam = cv2.VideoCapture(0)

# YOLO 가중치 파일과 CFG 파일 로드
YOLO_net = cv2.dnn.readNet("yolov3_final.weights", "yolov3_ct.cfg")
# YOLO NETWORK 재구성
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

frames = 0
start = time.time()

# 카메라 정상적으로 작동하는지 확인
if cam.isOpened() :
    ret, frame = cam.read()
    print('width :%d, height : %d' % (cam.get(3), cam.get(4)))
    print('카메라 정상작동')
else: # 실패시 false반환
    ret = False
    print('카메라 오류')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('cam_yolo_result.avi',fourcc, 20.0, (640,360))

while (ret):
    # 웹캠 프레임
    ret, frame = cam.read()
    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    #변수
    class_ids = []
    confidences = []
    boxes = []
    middle = []
    In_map_target_c=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w = int(detection[2] * w)
                h = int(detection[3] * h)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                middle.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            # 경계상자와 클래스 정보 투영
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

            # 객체 이미지 좌표값
            center = (((2 * x + w) / 2), ((2 * y + h) / 2))
            middle[i] = center

    # middle 값 저장 상태 확인
    for i in range(len(boxes)):
        if i in indexes:
            if i == 0 or (i - 1) % 3 == 0:
                if i == 0:
                    n = 1.0
                    print(n, "번째 타겟 영상/이미지 내 위치 : ", middle[i])
                if not i == 0:
                    n = (i - 1) / 3 + 1
                    print(n, "번째 타겟 영상/이미지 내 위치 : ", middle[i])

                print()
                # image 내부 계산 code
                target_f = middle[i]
                dy = target_f[1];
                dx = math.sqrt((w / 2 - target_f[0]) ** 2 + dy ** 2)
                az_radian = (drone_direction * (math.pi/180)) - math.atan2(dx, dy)
                print("이미지내 꺽인 (라이안)각도 : ", az_radian)
                az_degrees = math.degrees(az_radian)
                print("이미지내 꺽인 (일반)각도 : ", az_degrees)

                drone_position(middle, drone_center, lider, drone_direction, az_radian)

    frout = cv2.flip(frame,0)
    output.write(frout)

    cv2.imshow("YOLOv3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frames += 1
    print(time.time() - start)
    print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

# Release everything if job is finished
cv2.destroyAllWindows()