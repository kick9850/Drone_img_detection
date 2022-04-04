import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# 이미지 읽고 전처리 + 이미지 크기 얻기 (bounding box 크기 복구를 위함)
def load_image_pixels(filename, shape):
    image = filename
    width, height, ch = image.shape

    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0

    image = expand_dims(image, 0)

    return image, width, height

# 모든 박스에 대해 thrshold 확률 이상의 모든 라벨을 찾아서 붙이기
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # 모든 박스에 대해
    for box in boxes:
        # 모든 라벨을 검사
        for i in range(len(labels)):
            # 해당 라벨일 확률이 threshold 이상이면 저장 (append)
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # 하나 찾더라도 break 하지 말고, 확률이 threshold 이상인 모든 라벨을 찾을 것
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
    data = filename

    #plot each box
    for i in range(len(v_boxes)):
        # 상자 dims
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        start_pt = (box.xmin, box.ymax)
        end_pt = (box.xmax, box.ymin)
        color = (255, 0, 0)
        thickness = 2

        # 그리기
        data = cv2.rectangle(data, start_pt, end_pt, color, thickness)
        # 좌측 상단에 라벨과 확률 표시
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
        #영상 저장
        frout = cv2.flip(frame, 0)
        output.write(frout)
        #영상 표현
        cv2.imshow("YOLOv3", frame)

mode = 1       #mode = 0 캠, mode = 1 영상
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

if mode == 0:
    #카메라셋팅
    cam = cv2.VideoCapture(0)
    output = cv2.VideoWriter('yolo_result.avi', fourcc, 20.0, (640, 480))

#비디오 셋팅
elif mode == 1:
    video_file = "go_yolo.mp4"
    cam = cv2.VideoCapture(video_file)
    output = cv2.VideoWriter('yolo_result.avi', fourcc, 20.0, (640,480))

# 카메라 정상적으로 작동하는지 확인
if cam.isOpened() :
    ret, frame = cam.read()
    print('카메라 정상작동')

else: # 실패시 false반환
    ret = False
    print('카메라 오류')

model = load_model('pretrained_yolov3_model.h5')
# 라벨 정보
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# 60% 확률 이상인 라벨만 검출
class_threshold = 0.6

# anchors 정의
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

while (ret):
    ret, frame = cam.read()    #캠사용
    frame = cv2.resize(frame, (640, 480))

    height, width, c = frame.shape
    image, image_w, image_h = load_image_pixels(frame, (width, height))

    yhat = model.predict(image)
    print([a.shape for a in yhat])

    boxes = list()
    for i in range(len(yhat)):
        # 격자 크기 마다의 검출 결과 해석
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, height, width)

    # 검출된 박스 크기 보정
    correct_yolo_boxes(boxes, image_h, image_w, height, width)
    # 동일 물체를 잡아낸 박스들 병합
    do_nms(boxes, 0.5)

    # 박스 정보 읽어오기
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # draw what we found
    draw_boxes(frame, v_boxes, v_labels, v_scores)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cv2.destroyAllWindows()