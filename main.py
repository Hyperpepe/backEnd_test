import base64
import json
import logging
import os
import re
import threading
import time

import cv2
import flask
import numpy as np
import onnxruntime
from flask import request

# 日志输出
logging.basicConfig(filename="log.txt", level=logging.DEBUG, format="%(asctime)s: %(message)s")
# flask服务
api = flask.Flask(__name__)

# 读取模型以及标签名称
names = []
session = onnxruntime.InferenceSession('./model/daozha.onnx')
with open("./model/class-daozha.names", 'r') as f:
    for line in f.readlines():
        names.append(line.strip())
# GPIO列表
gpios = [
    "/proc/rp_gpio/gpioa12",  # A红
    "/proc/rp_gpio/gpioa11",  # A绿
    "/proc/rp_gpio/gpioao11",  # B红
    "/proc/rp_gpio/gpioa10",  # B绿
    "/proc/rp_gpio/gpioa0",  # C红
    "/proc/rp_gpio/gpioa13",  # C绿
]
gpioes = [
    "/proc/rp_gpio/gpioz6",  # A相分
    "/proc/rp_gpio/gpioz5",  # A相分
    "/proc/rp_gpio/gpioz4",  # B相合
    "/proc/rp_gpio/gpioz1",  # B相分
    "/proc/rp_gpio/gpioz0",  # C相合
    "/proc/rp_gpio/gpioz13",  # C相分
]


def control_gpio(gpio_index, value, classes):
    # Check if the value is valid
    if value != 0 and value != 1:
        print("Error: Invalid value")

        return
    # Get the path of the GPIO
    if classes == 1:
        gpio = gpios[gpio_index - 1]
        if gpio_index < 0 or gpio_index >= len(gpios) + 1:
            print("Error: Invalid GPIO index")
            return
    elif classes == 0:
        if gpio_index < 0 or gpio_index >= len(gpioes) + 1:
            print("Error: Invalid GPIO index")
            return
        gpio = gpioes[gpio_index - 1]
    elif classes == 2:
        for gpio in gpioes:
            os.system('echo 0 > ' + gpio)
    elif classes == 3:
        for gpio in gpios:
            os.system('echo 0 > ' + gpio)
    os.system('echo ' + str(value) + ' > ' + gpio)
    print("GPIO", gpio, "is now", value)
    logging.debug("GPIO" + gpio + "is now" + str(value))


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
        self.t1 = time.time()
        self.data = data
        self.result = None
        self.input_width = w
        self.input_height = h

    def run(self):
        # pic_info = self.data.get('picinfo')
        # self.pic_base64 = pic_info
        base64_code = re.sub('^data:image/.+;base64,', '', self.data)
        self.image_data = base64.b64decode(base64_code)
        pic_array = np.frombuffer(self.image_data, np.uint8)
        self.pic_array = cv2.imdecode(pic_array, cv2.IMREAD_UNCHANGED)
        self.results = detection(session, self.pic_array, self.input_width, self.input_height, 0.65)
        self.dataAnalyse()
        # self.get_result()

    # 解析检测结果
    def dataAnalyse(self):
        for b in self.results:
            print(b)
            self.obj_score, self.cls_index = b[4], int(b[5])
        self.name = names[self.cls_index]
        self.t2 = time.time()
        T = self.t2 - self.t1
        logging.debug("checkAI result : {} time : {}:".format(self.name, T))
        print("checkAI result : {} time : {}:".format(self.name, T))

    # 输出结果
    def get_result(self):
        return self.name


@api.route('/test', methods=['post'])
def test():
    ren = {'status': 'OK', 'status_code': 200}
    print(ren)
    logging.debug(ren)
    return json.dumps(ren, ensure_ascii=False)


@api.route('/checkleds', methods=['post'])
def checkleds():
    try:
        data = request.get_json()
        num = data['number']
        act = data['action']
        if num != 7:
            control_gpio(num, act, 1)
        else:
            control_gpio(num, act, 3)
        print('inputnum:', num, 'inputact:', act)
        logging.debug('inputnum: {} inputact: {}'.format(num, act))
        ren = {'status': 'OK', 'status_code': 200}
    except:
        ren = {'status': 'ERROR', 'status_code': 404}
    return json.dumps(ren, ensure_ascii=False)


@api.route('/checkrelay', methods=['post'])
def checkrelay():
    try:
        data = request.get_json()
        num = data['number']
        act = data['action']
        if num != 7:
            control_gpio(num, act, 0)
        else:
            control_gpio(num, act, 2)
        print('inputnum:', num, 'inputact:', act)
        logging.debug('inputnum: {} inputact: {}'.format(num, act))
        ren = {'status': 'OK', 'status_code': 200}
    except:
        ren = {'status': 'ERROR', 'status_code': 404}
    return json.dumps(ren, ensure_ascii=False)


@api.route('/checkAI', methods=['post'])
def checkAI():
    try:
        data = request.get_json()
        num = data['number']
        act = data['action']
        picinfo = data['picinfo']
        picin = PicInfo(picinfo, 352, 352)
        picin.start()
        picin.join()
        result = picin.name
        print(result)
        ren = {'status': result, 'status_code': 200}
        logging.debug("checkAI succeeded with number: {} and action: {}".format(num, act))
    except Exception as e:
        ren = {'status': 'ERROR', 'status_code': 404}
        logging.error("checkAI failed with error: {}".format(e))
    return json.dumps(ren, ensure_ascii=False)


if __name__ == '__main__':
    api.run(port=5000, debug=True, host='0.0.0.0')
