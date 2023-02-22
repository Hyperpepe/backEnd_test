import base64
import json
import re
import threading

import cv2
import flask
import numpy as np
import onnxruntime
from flask import request

api = flask.Flask(__name__)
session = onnxruntime.InferenceSession('./model/daozha.onnx')


# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# tanh函数
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1


# 数据预处理
def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255
    return output.astype('float32')


# nms算法
def nms(dets, thresh=0.4):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    print("dets.shape:", dets.shape)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标

    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)

        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output


# 目标检测
def detection(session, img, input_width, input_height, thresh):
    pred = []

    # 输入图像的原始宽高
    H, W, _ = img.shape

    # 数据预处理: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # 模型推理
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # 输出特征图转置: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # 输出特征图的宽高
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # 特征图后处理
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # 解析检测框置信度
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            # 阈值筛选
            if score > thresh:
                # 检测框类别
                cls_index = np.argmax(data[5:])
                # 检测框中心点偏移
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # 检测框归一化后的宽高
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # 检测框归一化后中心点
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height

                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                pred.append([x1, y1, x2, y2, score, cls_index])
    return nms(np.array(pred))


class PicInfo(threading.Thread):
    def __init__(self, data, w, h):
        threading.Thread.__init__(self)
        self.data = data
        self.result = None
        self.input_width = w
        self.input_height = h

    def run(self):
        pic_info = self.data.get('picinfo')
        self.pic_base64 = pic_info
        base64_code = re.sub('^data:image/.+;base64,', '', pic_info)
        self.image_data = base64.b64decode(base64_code)
        pic_array = np.frombuffer(self.image_data, np.uint8)
        self.pic_array = cv2.imdecode(pic_array, cv2.IMREAD_UNCHANGED)
        self.results = detection(session, self.pic_array, self.input_width, self.input_height, 0.65)
        names = []
        with open("../model/class-daozha.names", 'r') as f:
            for line in f.readlines():
                names.append(line.strip())
        print("result:" + str(imgname))

    def get_result(self):
        return self.result


@api.route('/test', methods=['post'])
def test():
    ren = {'status': 'OK', 'status_code': 200}
    print('/test')
    return json.dumps(ren, ensure_ascii=False)


@api.route('/checkleds', methods=['post'])
def checkleds():
    data = request.get_json()
    num = data['number']
    act = data['action']
    print(num, act)
    ren = {'status': 'OK', 'status_code': 200}
    return json.dumps(ren, ensure_ascii=False)


@api.route('/checkrelay', methods=['post'])
def checkrelay():
    data = request.get_json()
    num = data['number']
    act = data['action']
    print(num, act)
    ren = {'status': 'OK', 'msg_code': 200}
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
    cv2.imwrite('output' + str(num) + '.jpg', date)
    print(date)
    ren = {'msg': 'ERROR_NONE_ARGS', 'msg_code': 404}
    return json.dumps(ren, ensure_ascii=False)


if __name__ == '__main__':
    api.run(port=5000, debug=True, host='192.168.137.1')
