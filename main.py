import json
import logging
import time

import flask
import onnxruntime
from flask import request

from utils import control_gpio, PicInfo, output

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
        # print('inputnum:', num, 'inputact:', act)
        # logging.debug('inputnum: {} inputact: {}'.format(num, act))
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
        # print('inputnum:', num, 'inputact:', act)
        # logging.debug('inputnum: {} inputact: {}'.format(num, act))
        ren = {'status': 'OK', 'status_code': 200}
    except:
        ren = {'status': 'ERROR', 'status_code': 404}
    return json.dumps(ren, ensure_ascii=False)

@api.route('/startRS485Service', methods=['post'])
def startRS485Service():
    try:
        #打开串口
        print("open serial ttys3")

    except Exception as e:
        # 报错
        print(e)


@api.route('/stopRS485Service', methods=['post'])
def stopRS485Service():
    try:
        #打开串口
        print("close serial ttys3")

    except Exception as e:
        # 报错
        print(e)



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

        num_mapping = {
            "ALL": "全部刀闸",
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
        if result == "Running":  # Running状态必定异常
            final_result = "异常"
        elif result != act_f:
            final_result = "无异常"
        else :
            final_result = "异常"
        # print(final_result,num_f,act_f,result)
        t2 = time.time()
        T = t2 - t1
        t = '{:.3f}s'.format(T)
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
