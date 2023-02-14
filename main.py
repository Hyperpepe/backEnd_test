import base64
import re

import flask
import argparse
import time
import cv2
import numpy as np
import flask
from flask import json,request
api = flask.Flask(__name__)


class FastestDet():
    def __init__(self, confThreshold=0.3, nmsThreshold=0.4):
        self.classes = list(map(lambda x: x.strip(), open('model/coco.names','r').readlines()))
        ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.inpWidth = 512
        self.inpHeight = 512
        self.net = cv2.dnn.readNet('model/FastestDet.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.H, self.W = 32, 32
        self.grid = self._make_grid(self.W, self.H)

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)

                # confidences.append(float(confidence))

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        print(indices)
        for i in indices:
            # print(indices)
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        # if len(indices) == 0:
            # nospark()
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        print(label)
        # port_open_recv()
        a = self.classes[classId]
        # if self.classes[classId] != 0:
            # spark()

        print("a=", a)
        print("b=", classId)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        # print(label)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        pred = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        pred[:, 3:5] = self.sigmoid(pred[:, 3:5])  ###w,h
        pred[:, 1:3] = (np.tanh(pred[:, 1:3]) + self.grid) / np.tile(np.array([self.W, self.H]),
                                                                     (pred.shape[0], 1))  ###cx,cy
        srcimg = self.postprocess(srcimg, pred)
        # print(self.net.getUnconnectedOutLayersNames())
        return srcimg

class PicInfo:
    def __init__(self, data):
        pic_info = data.get('picinfo')
        self.pic_base64 = pic_info
        base64_code = re.sub('^data:image/.+;base64,', '', pic_info)
        self.image_data = base64.b64decode(base64_code)
        pic_array = np.frombuffer(self.image_data, np.uint8)
        self.pic_array = cv2.imdecode(pic_array, cv2.IMREAD_UNCHANGED)
        model = FastestDet(confThreshold=0.7, nmsThreshold=0.4)
        self.srcimg = model.detect(self.pic_array)

    def get_result(self):
        return self.srcimg
@api.route('/test', methods=['post'])
def test():
    ren = {'msg': 'OK', 'msg_code': 101}
    print('/test')
    return json.dumps(ren, ensure_ascii=False)

@api.route('/checkleds', methods=['post'])
def checkleds():
    data = request.get_json()
    num = data['number']
    act = data['action']
    print(num, act)
    ren = {'msg': 'OK', 'msg_code': 101}
    return json.dumps(ren, ensure_ascii=False)

@api.route('/checkrelay', methods=['post'])
def checkrelay():
    data = request.get_json()
    num = data['number']
    act = data['action']
    print(num,act)
    ren = {'msg': 'OK', 'msg_code': 101}
    return json.dumps(ren, ensure_ascii=False)

@api.route('/checkAI', methods=['post'])
def checkAI():
    data = request.get_json()
    num = data['number']
    act = data['action']
    picinfo = data['picinfo']
    picin = PicInfo(data)
    date = picin.get_result()
    # cv2.imshow('Detection Results', date)
    cv2.imwrite('output'+ str(num) +'.jpg', date)

    print(date)
    ren = {'msg': 'ERROR_NONE_ARGS', 'msg_code': 404}
    return json.dumps(ren, ensure_ascii=False)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--confThreshold', default=0.8, type=float, help='class confidence')
    # parser.add_argument('--nmsThreshold', default=0.35, type=float, help='nms iou thresh')
    # args = parser.parse_args()
    # model = FastestDet(confThreshold=0.7, nmsThreshold=0.4)
    # while True:
    #     time.sleep(1)
    #     start = time.time()
    #     cap = cv2.VideoCapture(0)
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #     success, img = cap.read()
    #     cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     srcimg = model.detect(img)
    #     cap.release()
    #     end = time.time()
    #     print("运行时间为：{}s".format(end - start))
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()

    api.run(port=5000, debug=True, host='0.0.0.0')