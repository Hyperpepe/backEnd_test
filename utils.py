import base64
import re
import socket
import cv2
import requests


# 检查转化后的图片是否可以读取
def check_image_valid(image_path):
    image = cv2.imread(image_path)
    if image is None:
        # 图片加载失败，不完整
        return False
    else:
        # 图片加载成功，完整
        return True


# 解析base64编码的图片
def base64_decode_jpg(picinfo, picname):
    base64_code = re.sub('^data:image/.+;base64,', '', picinfo)
    image_data = base64.b64decode(base64_code)
    filename = './picinfo/' + 'image' + picname + '.png'
    try:
        with open(filename, 'wb') as f:
            f.write(image_data)
            ret = check_image_valid(filename)
            # print("图片可以读取")
            if ret:
                print("图片" + picname + "可以读取")
                return True
            else:
                print("图片" + picname + "不可以读取")
    except Exception:
        return picname


# 结果请求
def request_post(ip_address, port, bodyin):
    host = "http://" + ip_address + ":" + port + "/"
    login_url = "/AICheckIsolaterjpg"
    url = host + login_url  # 拼接地址
    r = requests.post(url=url, json=bodyin)
    print(r.text)


# ping 通接受post的IP地址以及端口
def ping_ipaddr_port(ipaddr, port):
    port = int(port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ipaddr, port))
        print('%s:%d 已开放' % (ipaddr, port))
        s.close()
        return True
    except socket.error as e:
        print('%s:%d 未开放' % (ipaddr, port))
        s.close()
        return False


# 检查ip以及端口是否可达，以便后续post使用
def check_args_ipaddress(data):
    # ret = False
    if data:
        operator = data['operator']
        listen_port_info = data.get('listenPortInfo')  # 读取 listenPortInfo 中的 IpAddress 和 Port
        if listen_port_info:
            ip_address = listen_port_info.get('IpAddress')
            port = listen_port_info.get('Port')
            if ip_address and port:
                retip = ping_ipaddr_port(ip_address, port)
                if retip:
                    print("IP 地址和端口可达")
                    ret = True
                else:
                    print("IP 地址和端口不可达")
                    ret = False
            else:
                print("IP 地址和端口信息不完整")
                ret = False
        else:
            print("listen_port_info 不完整")
            ret = False
    else:
        print("data不完整")
        ret = False
    return ret


# 检查IsolaterInfo是否完整且能够读取
def check_args_IsolaterInfo(data):
    isolater_info = data.get('IsolaterInfo')  # 读取 IsolaterInfo 中的 issueNumber、IsolaterID、IsolaterName 和 IsolaterCmd
    if isolater_info:
        print("取到IsolaterInfo整个info")
        issue_number = isolater_info.get('issueNumber')
        isolater_id = isolater_info.get('IsolaterID')
        isolater_name = isolater_info.get('IsolaterName')
        isolater_cmd = isolater_info.get('IsolaterCmd')
        if all(value for value in isolater_info.values()):
            print("IsolaterInfo 中所有值都有值")
            if (isolater_cmd == "1" or isolater_cmd == "0"):
                print("isolater_cmd 是数字 1 或 0")
                ret = True
            else:
                print("isolater_cmd 不是数字 1 或 0")
                ret = False
        else:
            print("IsolaterInfo 中有值为空")
            ret = False
    else:
        print("取不到整个info")
        ret = False
    return ret


# 检查picinfo能否转化为可读取的图片
def check_args_picinfo(data):
    # global ret
    pic_info = data.get('picinfo')  # 读取 picinfo 中的 APicInfo、BPicInfo 和 CPicInfo
    if pic_info:
        a_pic_info = pic_info.get('APicInfo')
        b_pic_info = pic_info.get('BPicInfo')
        c_pic_info = pic_info.get('CPicInfo')
        # write_json_data_to_file(a_pic_info)
        if a_pic_info and b_pic_info and c_pic_info:
            reta = base64_decode_jpg(a_pic_info, 'APicInfo')
            retb = base64_decode_jpg(b_pic_info, 'BPicInfo')
            retc = base64_decode_jpg(c_pic_info, 'CPicInfo')
            if reta and retc and retb:
                print("base64图片检查——以下参数符合条件:", reta, retb, retc)
                ret = True
            else:
                print("base64图片检查——以下参数不符合条件:", reta, retb, retc)
                ret = False
    else:
        print("pic_info 不完整")
        ret = False
    return ret
