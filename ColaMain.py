from bisect import bisect
from threading import Thread
from time import sleep
import cv2
import numpy as np
import socket
import json

# Cola import
from ColaHands import ColaHands
from ColaPose import ColaPose
from ColaActionExplanation import ColaActionExplanation
from ColaProjectUtils import cola_get_conf

args = cola_get_conf(ip_self="127.0.0.1:4321", ip_target="127.0.0.1:4322")
# args = cola_get_conf(ip_self="192.168.43.244:4321", ip_target="192.168.43.172:4322")


class ColaMain:

    def __init__(self, id_caps=[0, 1]) -> None:
        # two camera
        self.cap1 = cv2.VideoCapture(id_caps[0])  # hands
        self.cap2 = cv2.VideoCapture(id_caps[1])
        print("Opening camera successful.")

        # model
        self.cola_hands = ColaHands(False)
        self.cola_pose = ColaPose(False)
        print("Creating model successful.")

        # udp socket
        self.udp_addr_self = args.s  # sefl
        self.udp_addr_target = args.t  # target
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(self.udp_addr_self)
        print("Creating socket successful.")

        # usefull data
        self.hands_data = (None, ) * 2  # hands_type, hands_rslt
        self.pose_data = (None, ) * 2  # pose_type, pose_rslt
        
        a = 100
        b = 123

        # system control
        self.is_system_running = False
        self.is_data_updated = False
        self.is_connected_data_stream = False

        # explain
        self.cola_action_exp = ColaActionExplanation()

        print("All done well!")

    def cap_release(self, cap):
        if cap:
            cap.release()
            cap = None

    def __del__(self):
        self.cap_release(self.cap1)
        self.cap_release(self.cap2)
        self.udp_socket.close()

    def run1(self):
        while self.cap1.isOpened() and self.cap2.isOpened():
            # img
            ret1, img1 = self.cap1.read()
            ret2, img2 = self.cap2.read()
            if not (ret1 and ret2):
                print("Camera error.")
                break

            self.hands_data = self.cola_hands.process(img1)
            self.pose_data = self.cola_pose.process(img2)

            # just show
            hands_type, hands_type_name, hands_rslt = self.hands_data
            pose_type, pose_type_name, pose_rslt = self.pose_data

            code = [pose_type, hands_type]
            print(code, ':\t', pose_type_name, ' + ', hands_type_name)

            # latency
            cv2.waitKey(100)

    def udp_receive(self):
        msg, ip_socket = self.udp_socket.recvfrom(1024)
        msg = msg.decode('utf-8')
        return msg if ip_socket == self.udp_addr_target else ""

    def udp_send(self, msg):
        """
        msg: utf-8 not bite
        """
        print("MSG: ", msg)
        self.udp_socket.sendto(msg.encode('utf-8'), self.udp_addr_target)

    def run(self):
        """
        thread1: control udp to receive message to control mode
        thread2: run model to change the value of hands_type, pose_type blabla ...
        principle: 
            1. return message: got it blabla...(by command)
        """
        mode_name_func = {
            # start system for action recognize and 3D skeloton recognition
            "CONTROL_SYSTEM_START":
            self.control_system_start,
            # system stanby -> all stop and waiting for control
            "CONTROL_SYSTEM_STANBY":
            self.control_system_stanby,
            # system state infomation
            "GET_SYSTEM_STATE":
            self.get_system_state,
            # system log infomation
            "GET_SYSTEM_LOG":
            self.get_system_log,
            # control for get recognition data stream
            "CONTROL_GET_RECOGNITION_DATA_STREAM":
            self.control_get_recognition_data_stream,
            # control for stop recognition data stream
            "CONTROL_STOP_RECOGNITION_DATA_STREAM":
            self.control_stop_recognition_data_stream
        }
        # other control: CONTROL_SYSTEM_BREAK_DOWN
        while True:
            recv_msg = self.udp_receive()
            if recv_msg == "CONTROL_SYSTEM_BREAK_DOWN":
                self.is_system_running = False
                break

            # switch msg
            print(recv_msg)
            if recv_msg in mode_name_func.keys():
                mode_name_func[recv_msg]()
            else:
                self.udp_send("Message error.")
            sleep(0.5)

    def control_system_start(self):
        """
        brief: create new thread to run recognition and update data
        """
        if self.is_system_running == True:
            self.udp_send("System already running.")
            return
        # set state
        self.is_system_running = True
        # run recognizer
        t_run = Thread(target=self.run_recognizer)
        t_run.start()
        # feed back
        self.udp_send("System started!")

    def control_system_stanby(self):
        if self.is_system_running == False:
            self.udp_send("System already stanby.")
        else:
            self.is_system_running = False
            self.udp_send("System stanby finished.")

    def control_stop_recognition_data_stream(self):
        if self.is_connected_data_stream == False:
            self.udp_send("Data stream already stoped!")
        else:
            self.is_connected_data_stream = False
            sleep(1)
            self.udp_send("Data stream stoped!")

    def get_message_from_recognition_data(self):
        # version for test
        # action_dict = {"pose": self.pose_data[0], "hands": self.hands_data[0]}
        # info = self.cola_action_exp.explain_to_string(action_dict)
        # data_json = json.dumps(action_dict)

        # version to extract  # TODO
        keys = ['pose', 'pose_rslt', "hands", 'hands_rslt']
        data_dict = zip(keys, (self.pose_data[0], self.pose_data[1][8:10],
                               self.hands_data[0], self.hands_data[1][8:10]))
        data_json = json.dumps(dict(data_dict))
        return data_json

    def control_get_recognition_data_stream(self):
        """
        constant send data to udp_target
        """
        self.control_system_start()
        if self.is_connected_data_stream:
            self.udp_send("Data stream already started.")
        self.is_connected_data_stream = True
        t_run = Thread(target=self._control_get_recognition_data_stream)
        t_run.start()

    def _control_get_recognition_data_stream(self):
        while self.is_system_running and self.is_connected_data_stream:
            # waiting for send
            if self.is_data_updated != True:
                continue
            self.is_data_updated = False

            # send message
            self.udp_send(self.get_message_from_recognition_data())

    def get_system_state(self):
        """
        send message to udp_target
        """
        message = "IS_RUNNING" if self.is_system_running else "NOT_RUNNING"
        self.udp_send(message)

    def get_system_log(self):
        self.udp_send("LOG blabla")

    def run_recognizer(self):
        while self.is_system_running:
            # img
            ret1, img1 = self.cap1.read()
            ret2, img2 = self.cap2.read()
            if not (ret1 and ret2):
                print("Camera error.")
                break
            # process data and TODO: update data
            self.hands_data = self.cola_hands.process(img1)
            self.pose_data = self.cola_pose.process(img2)

            # just show
            # hands_type_list, hands_rslt = self.hands_data
            # pose_type_list, pose_rslt = self.pose_data

            # show data
            self.is_data_updated = True

            # latency
            cv2.waitKey(100)


if __name__ == "__main__":
    colaer = ColaMain()
    colaer.run()
