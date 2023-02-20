import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Cola import
from ColaMediaPipeUtils import (landmark_to_ndarray, landmark_to_list)
from ColaUtils import (dot_product_angle, fig2data, scale_img_with_cons_ration)
from ColaActionExplanation import ColaActionExplanation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# algorithm to recognize action
class CodeRecognitioner:

    def __init__(self) -> None:
        self.code = None
        self.code_list = [0] * 20  # just for visulization

        self.times_forward_180 = 0
        self.times_forward_90 = 0

        # for type 3
        self.times_type_3 = 0
        self.curr_state = 0
        
        # for type 4
        self.times_type_4 = 0
        self.curr_state_4 = 0

    def append(self, code):
        times_threash = 2
        self.code = code
        self.code_list.pop(0)
        self.code_list.append(code)
        # memorization it with time
        if code == [2, 1]:
            self.times_forward_180 += 1
        else:
            self.times_forward_180 = 0

        # for type 3
        if self.curr_state != 0:
            self.times_type_3 += 1
        if self.times_type_3 > 30:
            self.times_type_3 = 0
            self.curr_state = 0

        if self.curr_state == 0 and code == [2, 1]: 
            self.curr_state = 1
        elif self.curr_state == 1 and code == [1, 1]:
            self.curr_state = 2
        elif self.curr_state == 2 and code == [2, 1]:
            self.curr_state = 3
            
        # check state
        type_id = 0
        if code == [2, 0]:
            self.times_forward_90 += 1
        else:
            self.times_forward_90 = 0
        
        if self.curr_state == 3:  # 手臂向前伸直90°上下舞动
            self.times_type_3 = 0
            self.curr_state = 0
            type_id = 3
        elif self.times_forward_180 >= times_threash:  # 手臂向前伸直
            type_id = 1
        elif self.times_forward_90 >= times_threash:  # 手臂向前伸直 && 手肘向上90°
            type_id = 2
        else:
            type_id = 0  # Default
            
        # for type 4
        if self.curr_state_4 != 0:
            self.times_type_4 += 1
        if self.times_type_4 > 30:
            self.times_type_4 = 0
            self.curr_state_4 = 0
            
        if type_id == 3:
            if self.curr_state_4 <= 1:
                self.curr_state_4 += 1
            else:
                self.curr_state_4 = 0
                self.times_type_4 = 0
                type_id = 4
        return [type_id]


class ColaPose:

    def __init__(self, is_using_cap=True) -> None:
        self.cap = cv2.VideoCapture(0) if is_using_cap else None
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.code_recog = CodeRecognitioner()
        self.cola_action_exp = ColaActionExplanation()

    def release_cap(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release_cap()

    def get_frame(self):
        return self.cap.read()

    def encode_pose_theta(self, rslt_processed):
        code = [0] * 2
        code[0] = bisect.bisect([30, 70, 140], rslt_processed[0])
        code[1] = int(rslt_processed[1] >= 110)  # 0: 0-100->弯曲 1: 直立
        return code

    def process(self, img):
        """
        img: BGR 
        return: rslt extracted from image
        """
        rslt = self.reference_pose(img)
        rslt_processed = self.processe_pose_rslt(rslt)

        # encode
        code = self.encode_pose_theta(rslt_processed)
        pose_type_list = self.code_recog.append(code)
        return pose_type_list, self.rslt_pose_reformat(rslt)

    def rslt_pose_reformat(self, rslt):
        data_list = []
        if not rslt.pose_landmarks:
            return []
        for lamdmark in rslt.pose_landmarks.landmark:
            data_list.append(landmark_to_list(lamdmark, is_vis=True))
        return data_list

    def run_recognition(self):
        while self.cap.isOpened():
            ret, img = self.get_frame()
            if not ret:
                print("Camera read error.")
                break
            rslt = self.reference_pose(img)  # 
            rslt_processed = self.processe_pose_rslt(rslt)

            # encode for recognition
            code = self.encode_pose_theta(rslt_processed)
            pose_type_list = self.code_recog.append(code)
            self.cola_action_exp.explain({"pose": pose_type_list})

            # wait
            cv2.waitKey(100)

    def run_visualize(self):
        t = int(0)
        fig = plt.figure()
        rslt_pro_list = []
        while self.cap.isOpened():
            ret, img = self.get_frame()
            if not ret:
                print("Camera read error.")
                break
            # reference
            rslt = self.reference_pose(img.copy())

            # # draw notation
            self.draw_pose_notation(img, rslt)
            rslt_processed = self.processe_pose_rslt(rslt)
            code = self.encode_pose_theta(rslt_processed)
            print(code)

            # save data
            rslt_pro_list.append(rslt_processed)

            # draw scatter
            plt.scatter(t, rslt_processed[0], color=[0, 0, 1.0])
            plt.scatter(t, rslt_processed[1], color=[0, 1.0, 0])
            plt.legend(labels=["theta_1", "theta_2"], loc='best')
            t += 1
            plt.xlim((t - 20, t + 1))
            plt.ylim((0, 180))
            img_data = fig2data(fig)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)

            # show img
            h, w, _ = img.shape
            img_data = scale_img_with_cons_ration(img_data, img_data.shape[1],
                                                  h)
            img_all = np.concatenate((img, img_data), axis=1)
            cv2.imshow("ColaWin", img_all)
            if cv2.waitKey(30) & 0xff == 27:
                break
        cv2.destroyAllWindows()

        # save theta data
        data = np.asarray(rslt_pro_list, dtype=np.float32)
        np.savetxt("./cola_store/pose_theta_data2.txt", data)

    def processe_pose_rslt(self, rslt):
        """
        rslt: output of model
        return: feature from the output
        """
        if not rslt.pose_landmarks:
            return [0, 0]
        kp_24, kp_12, kp_14, kp_16 = tuple(
            landmark_to_ndarray(rslt.pose_landmarks.landmark[i])
            for i in [24, 12, 14, 16])
        theta1 = dot_product_angle(kp_24[0:2] - kp_12[0:2],
                                   kp_14[0:2] - kp_12[0:2])
        theta2 = dot_product_angle(kp_16[0:2] - kp_14[0:2],
                                   kp_12[0:2] - kp_14[0:2])
        # print(theta1, theta2)
        return [theta1, theta2]

    def draw_pose_notation(self, img, rslt):
        mp_drawing.draw_landmarks(img,
                                  rslt.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.
                                  get_default_pose_landmarks_style())

    def reference_pose(self, img):
        """
        img: BGR image
        return: output of the model
        """
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rslt = self.pose.process(img)
        return rslt


if __name__ == "__main__":
    # version 1
    cola_pose = ColaPose(True)
    # cola_pose.run_visualize()
    cola_pose.run_recognition()

    # version 2
    # cola_pose = ColaPose()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, img = cap.read()
    #     if not ret:
    #         print("Image reading error.")
    #         break
    #     pose_type, pose_data = cola_pose.process(img)
    #     print(pose_type, pose_data[0] if pose_data != [] else None)
    #     cv2.waitKey(100)