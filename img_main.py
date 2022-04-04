import cv2
import numpy as np
import math

#레이더 임의 거리 값
lider = 100;

#드론 좌표 임의값
drone_x, drone_y, drone_z = 10, 10, 10;
drone_center = (drone_x, drone_y, drone_z);
drone_direction = 10;

# Yolo 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 가져오기
img = cv2.imread("image/sample.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
middle = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            middle.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # 객체 이미지 좌표값
        center = (((2*x+w) / 2), ((2*y+h) / 2))
        middle[i] = center

#middle 값 저장 상태 확인
for i in range(len(boxes)):
    if i in indexes:
        if i == 0 or (i-1)%3 == 0:
            if i == 0:
                n = 1.0
                print(n, "번째 타겟 영상/이미지 내 위치 : ", middle[i])
            else:
                n = (i - 1)/3 + 1
                print(n, "번째 타겟 영상/이미지 내 위치 : ", middle[i])
print()
#image 내부 계산 code
target_f = middle[0]
dy = target_f[1];
dx = math.sqrt((width/2 - target_f[0])**2 + dy**2)
az_radian = math.acos(dy/dx)
print("이미지내 꺽인 (라이안)각도 : ",az_radian)

az_degrees=math.degrees(az_radian)
print("이미지내 꺽인 (일반)각도 : ",az_degrees)

#실제 환경 계산 code
print()
#드론중심으로 타겟의 정보
In_map_target_st = math.sqrt(lider**2 - drone_y**2)
In_map_target_dy = drone_y

In_map_target_az = math.acos(drone_y/lider)
print("드론 타겟 사이 꺽인 (라디안)각도 : ", In_map_target_az)
print("드론 타겟 사이 (일반)각도 : ", math.degrees(In_map_target_az))
print("실제 드론과 타겟의 거리  : ", In_map_target_st)

print()
#맵 중심의 타켓 정보
print("맵상 드론이 바라보는 각도 : ",drone_direction)
#드론 정보
print("드론 위치 : ", drone_center)

# xy : 맵상 좌표 / z: 높이
In_map_target_cx = drone_center[0] + In_map_target_st * math.cos(math.radians(drone_direction)+In_map_target_az)
In_map_target_cy = drone_center[1] + In_map_target_st * math.sin(math.radians(drone_direction)+In_map_target_az)
In_map_target_cz = drone_center[2]

In_map_target_c = (In_map_target_cx,In_map_target_cy,In_map_target_cz)
print("타켓 예상 위치 : ", In_map_target_c)

cv2.imwrite('image/test.jpg', img)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()