from inspect import FrameInfo
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Cola import
from ColaMediaPipeUtils import (landmark_to_ndarray, landmark_to_list)
from ColaUtils import dot_product_angle
from ColaActionExplanation import ColaActionExplanation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class ColaHands:

    def __init__(self, is_using_camera=False) -> None:
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0) if is_using_camera else None

        self.table_theta_range = [[60, 100], [60, 176], [50, 180], [55, 200],
                                  [53, 208]]
        self.cola_action_exp = ColaActionExplanation()

    def release_cap(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release_cap()

    def get_frame(self):
        return self.cap.read()

    def code_to_type(self, code):
        if code[1:5] == [0] * 4 and code[0] < 2:
            type_hands = 1  # "五指伸直"
        elif code[1:5] == [2] * 4 and code[0] < 2:
            type_hands = 2  # "大拇指竖起"
        elif code[0] == 0 and code[1:4] >= [1]*3 and code[4] <= 1:
            type_hands = 3  # "大/小指母翘起"
        elif code[1:3] == [
                0
        ] * 2 and code[0] >= 1 and code[3] >= 1 and code[4] >= 1:
            type_hands = 4  # "二指并拢"
        elif code == [2] * 5:
            type_hands = 5  # "单手握拳"
        else:
            type_hands = 0  # "默认"
        return [type_hands]

    def process(self, img):
        """
        img: BGR from opencv
        """
        rslt = self.reference_hands(img.copy())  # reference
        rslt_processed = self.processe_hands_rslt(rslt)  # process data
        code = self.encode_hands_theta(rslt_processed)
        hands_type = self.code_to_type(code)
        rslt_dict = self.rslt_hands_reformat(rslt)
        return hands_type, rslt_dict

    def rslt_hands_reformat(self, rslt):
        """
        return data that from processed
        """
        # version1: return the degree of each five figures: list of [d1, d2, d3, d4, d5]
        rslt_value_list = []
        # landmarks_ndarray = np.ndarray(shape=(21, 3), dtype=np.float32)
        if rslt.multi_hand_landmarks:
            for value_lm in rslt.multi_hand_landmarks[
                    0].landmark:  # TODO: first hands
                rslt_value_list.append(landmark_to_list(value_lm))
        return rslt_value_list

    def run_visualize(self):
        theta_data_list = []
        while self.cap.isOpened():
            ret, img = self.get_frame()
            if not ret:
                print("Image reading error.")
                break

            # reference
            rslt = self.reference_hands(img.copy())

            # draw notation
            self.draw_hands_notation(img, rslt)
            rslt_processed = self.processe_hands_rslt(rslt)
            # print(rslt_processed)

            # encoding
            code = self.encode_hands_theta(rslt_processed)
            # print(code)
            hands_type_list = self.code_to_type(code)
            self.cola_action_exp.explain({'hands': hands_type_list})

            # show image
            cv2.imshow("ColaWin", img)
            if cv2.waitKey(100) & 0xff == 27:
                break
        np.savetxt("./cola_store/hands_theta_data2.txt",
                   np.asarray(theta_data_list))

    def encode_hands_theta(self, rslt_processed):
        code = [-1] * 5
        if rslt_processed is None:
            return code
        else:
            for idx_fig in range(5):
                code[idx_fig] = bisect.bisect(self.table_theta_range[idx_fig],
                                              rslt_processed[idx_fig])
        return code

    def handness_to_label(self, cl):
        cl_s = cl.__str__()
        idx1 = cl_s.find('label: ') + len("label: ") + 1
        idx2 = cl_s.find("\n}") - 1
        return cl_s[idx1:idx2]

    def processe_hands_rslt(self, rslt):
        """
        return data that from processed
        """
        if rslt.multi_hand_landmarks:
            degree_figures = np.zeros(shape=(5, ), dtype=np.float32)
            for hand_landmarks, handness in zip(
                    rslt.multi_hand_landmarks,
                    rslt.multi_handedness):  # each hand
                if self.handness_to_label(handness) != 'Left':
                    continue
                kp_0 = landmark_to_ndarray(hand_landmarks.landmark[0])
                for idx_fig in range(5):
                    kp_1, kp_2, kp_3, kp_4 = tuple(
                        landmark_to_ndarray(hand_landmarks.landmark[idx])
                        for idx in (idx_fig * 4) + np.asarray([1, 2, 3, 4]))
                    theta1 = dot_product_angle(kp_0 - kp_1, kp_2 - kp_1)
                    theta2 = dot_product_angle(kp_1 - kp_2, kp_3 - kp_2)
                    theta3 = dot_product_angle(kp_2 - kp_3, kp_4 - kp_3)
                    degree_figures[idx_fig] = 180 * 3 - (theta1 + theta2 +
                                                         theta3)
                return degree_figures.tolist()
        else:
            return None

    def reference_hands(self, img):
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(img)

    def draw_hands_notation(self, img, rslt):
        if rslt.multi_hand_landmarks:
            for hand_landmarks in rslt.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


if __name__ == "__main__":
    # test1
    cola_hands = ColaHands(True)
    cola_hands.run_visualize()

    # test2
    # cola_hands = ColaHands()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, img = cap.read()
    #     if not ret:
    #         print("Image reading error.")
    #         break
    #     hands_type, hands_data = cola_hands.process(img)
    #     print(hands_type, hands_data[0] if hands_data != [] else None)
