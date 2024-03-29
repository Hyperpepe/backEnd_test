import os
import sys
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
# noinspection PyPackageRequirements
import base64,json,logging,re,threading,time,serial,cv2,flask
import numpy as np
from flask import request
import onnxruntime

# 日志输出
logging.basicConfig(filename="log.txt", level=logging.DEBUG, format="%(asctime)s: %(message)s")
# flask服务
api = flask.Flask(__name__)

# 读取模型以及标签名称
session = onnxruntime.InferenceSession('daozha.onnx')
with open("class-daozha.names", 'r') as f:
    names = []
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
    "/proc/rp_gpio/gpioz13",  # A相分
    "/proc/rp_gpio/gpioz4",  # A相合
    "/proc/rp_gpio/gpioz0",  # B相分
    "/proc/rp_gpio/gpioz5",  # B相合
    "/proc/rp_gpio/gpioz1",  # C相分
    "/proc/rp_gpio/gpioz6",  # C相和
]

def output(num, act, result):
    # control_gpio(7, 0, 3)
    # control_gpio(7, 0, 2)
    if num == "ALL":  # A,B,C 相
        if act == "C":
            if result == "Opened":
                gpioled = [gpios[1], gpios[3], gpios[5]]
                gpiorelay = [gpioes[1], gpioes[3], gpioes[5]]
                # 全部置高
                for gpiol in gpioled:
                    os.system('echo 1 > ' + gpiol)
                for gpior in gpiorelay:
                    os.system('echo 1 > ' + gpior)
            elif result == "Closed":
                # led全灭
                # 继电器全开
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Running":
                # led全灭
                # 继电器全开
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
        elif act == "O":
            if result == "Opened":
                # led全灭
                # 继电器全开
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Closed":
                gpioled = [gpios[0], gpios[2], gpios[4]]
                gpiorelay = [gpioes[0], gpioes[2], gpioes[4]]
                for gpiol in gpioled:
                    os.system('echo 0 > ' + gpiol)
                for gpior in gpiorelay:
                    os.system('echo 0 > ' + gpior)
            elif result == "Running":
                # led全灭
                # 继电器全开
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
    elif num == "A":  # A 相
        if act == "C":
            if result == "Opened":  # 开到位
                os.system('echo 0 > ' + gpios[0])
                os.system('echo 1 > ' + gpios[1])
                os.system('echo 0 > ' + gpioes[0])
                os.system('echo 1 > ' + gpioes[1])
                # print('echo 1 > ' + gpios[1])
                # print('echo 1 > ' + gpioes[0])
            elif result == "Closed":  # 开不到位
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Running":  # 开不到位
                print("ALL relay are reset", 'output')
                print("ALL leds are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
        elif act == "O":
            if result == "Opened":  # 合不到位
                print("ALL relay are reset", 'output')
                print("ALL leds are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Closed":  # 合到位
                os.system('echo 1 > ' + gpios[0])
                os.system('echo 0 > ' + gpios[1])
                os.system('echo 1 > ' + gpioes[0])
                os.system('echo 0 > ' + gpioes[1])
                print('echo 1 > ' + gpios[0])
                print('echo 1 > ' + gpioes[1])
            elif result == "Running":  # 合不到位
                print("ALL relay are reset", 'output')
                print("ALL leds are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
    elif num == "B":  # B 相
        if act == "C":
            if result == "Opened":
                os.system('echo 0 > ' + gpios[2])
                os.system('echo 1 > ' + gpios[3])
                os.system('echo 0 > ' + gpioes[2])
                os.system('echo 1 > ' + gpioes[3])
                print('echo 1 > ' + gpios[3])
                print('echo 1 > ' + gpioes[2])
            elif result == "Closed":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Running":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
        elif act == "O":
            if result == "Opened":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Closed":
                os.system('echo 1 > ' + gpios[2])
                os.system('echo 0 > ' + gpios[3])
                os.system('echo 1 > ' + gpioes[2])
                os.system('echo 0 > ' + gpioes[3])
                print('echo 1 > ' + gpioes[3])
                print('echo 1 > ' + gpios[2])
            elif result == "Running":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
    elif num == "C":  # C 相
        if act == "C":
            if result == "Opened":
                os.system('echo 0 > ' + gpios[4])
                os.system('echo 1 > ' + gpios[5])
                os.system('echo 0 > ' + gpioes[4])
                os.system('echo 1 > ' + gpioes[5])
                print('echo 1 > ' + gpios[5])
                print('echo 1 > ' + gpioes[4])
            elif result == "Closed":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Running":
                print("ALL leds are reset", 'output')

                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
        elif act == "O":
            if result == "Opened":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)
            elif result == "Closed":
                os.system('echo 1 > ' + gpios[4])
                os.system('echo 0 > ' + gpios[5])
                os.system('echo 1 > ' + gpioes[4])
                os.system('echo 0 > ' + gpioes[5])
                print('echo 1 > ' + gpios[4])
                print('echo 1 > ' + gpioes[5])
            elif result == "Running":
                print("ALL leds are reset", 'output')
                print("ALL relay are reset", 'output')
                control_gpio(7, 0, 3)
                control_gpio(7, 0, 2)





# def output(num, act, result):
#     def control_leds(leds, values):
#         for led, value in zip(leds, values):
#             os.system(f'echo {value} > {led}')
#
#     def control_relays(relays, values):
#         for relay, value in zip(relays, values):
#             os.system(f'echo {value} > {relay}')
#
#     def reset_all():
#         control_gpio(7, 0, 3)
#         control_gpio(7, 0, 2)
#
#     if num == "ALL":
#         if act == "C":
#             if result == "Opened":
#                 control_leds([gpios[1], gpios[3], gpios[5]], [1, 1, 1])
#                 control_relays([gpios[0], gpios[2], gpios[4]], [1, 1, 1])
#             else:  # "Closed" or "Running"
#                 reset_all()
#         elif act == "O":
#             if result == "Closed":
#                 control_leds([gpios[0], gpios[2], gpios[4]], [0, 0, 0])
#                 control_relays([gpios[1], gpios[3], gpios[5]], [0, 0, 0])
#             else:  # "Opened" or "Running"
#                 reset_all()
#     else:
#         idx = {"A": 0, "B": 2, "C": 4}[num]
#         if act == "C":
#             if result == "Opened":
#                 control_leds([gpios[idx + 1]], [1])
#                 control_relays([gpios[idx], gpioes[idx]], [1, 0])
#             else:  # "Closed" or "Running"
#                 reset_all()
#         elif act == "O":
#             if result == "Closed":
#                 control_leds([gpios[idx]], [1])
#                 control_relays([gpios[idx + 1], gpioes[idx + 1]], [0, 1])
#             else:  # "Opened" or "Running"
#                 reset_all()


def control_gpio(gpio_index, value, classes):
    # Check if the value is valid
    # global gpio
    global gpio
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
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "ALL leds are reset", "control_gpio")
        for gpio in gpioes:
            os.system('echo 0 > ' + gpio)

    elif classes == 3:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "ALL relays are reset", "control_gpio")
        for gpio in gpios:
            os.system('echo 0 > ' + gpio)

    os.system('echo {0} > {1}'.format(str(value), gpio))
    print("GPIO", gpio, "is now", value)
    # logging.debug("GPIO" + gpio + "is now" + str(value))



# def control_gpio(gpio_index, value, classes):
#     # Check if the value is valid
#     if value not in (0, 1):
#         print("Error: Invalid value")
#         return
#
#     # Helper function to control GPIO pins
#     def set_gpio_pins(pin_list, pin_value):
#         for pin in pin_list:
#             os.system(f'echo {pin_value} > {pin}')
#             print(f"GPIO {pin} is now {pin_value}")
#
#     if classes in (0, 1):
#         # Control individual GPIO pin
#         gpio_list = gpios if classes == 1 else gpioes
#
#         if 0 <= gpio_index < len(gpio_list) + 1:
#             gpio = gpio_list[gpio_index]
#             os.system(f'echo {value} > {gpio}')
#             print(f"GPIO {gpio} is now {value}")
#         else:
#             print("Error: Invalid GPIO index")
#
#     elif classes == 2:
#         # Reset all LED states
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "ALL leds are reset", "control_gpio")
#         set_gpio_pins(gpioes, 0)
#
#     elif classes == 3:
#         # Reset all relay states
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "ALL relays are reset", "control_gpio")
#         set_gpio_pins(gpios, 0)
control_gpio(7, 0, 3)
control_gpio(7, 0, 2)



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

    # 数据预处理: resize, 1/25v5
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


ser = serial.Serial('/dev/ttyS3', 9600, timeout=0.001)


def echo_serial():
    global ser
    while True:
        data = ser.readline().strip()
        ser.write(data)


ser_thread = threading.Thread(target=echo_serial)
ser_thread.start()

# control_gpio(7, 0, 3)
# control_gpio(7, 0, 2)


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
        final_result = None
        t1 = time.time()
        data = request.get_json()
        num = data['number']
        act = data['action']
        picinfo = data['picinfo']
        picin = PicInfo(picinfo, 352, 352)
        picin.start()
        picin.join()
        result = picin.name
        print(num, act)
        output(num, act, result)
        t2 = time.time()
        T = t2 - t1
        t = '{:.3f}s'.format(T)
        num_mapping = {
            "ALL": "ALL",
            "A": "A相",
            "B": "B相",
            "C": "C相"
        }
        act_mapping = {
            "C": "Closed",
            "O": "Opened",
        }
        act_f = act_mapping.get(act, None)
        num_f = num_mapping.get(num, None)
        if result == "Running" or result == act_f:  # Running状态必定异常
            final_result = "异常"
        elif result != act_f:
            final_result = "无异常"
        ren = {
            'num': num_f,
            'result': final_result,
            'status': result,
            'status_code': 200,
            'time': t
        }

    except Exception as e:
        ren = {'status': 'ERROR', 'status_code': 404}
        logging.error("checkAI failed with error: {}".format(e))
    return json.dumps(ren, ensure_ascii=False)


if __name__ == '__main__':
    api.run(port=5000, debug=True, host='0.0.0.0')
    control_gpio(7, 0, 3)
    control_gpio(7, 0, 2)
