# 兼职1: 实现人体右手举手的识别、次数统计
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import bisect
import argparse
from typing import List, Mapping, Optional, Tuple, Union
import math

# Cola import
from ColaMediaPipeUtils import (landmark_to_ndarray, landmark_to_list)
from ColaUtils import (dot_product_angle, fig2data, scale_img_with_cons_ration)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Translate landmark to pixel
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def landmark_to_pixel(landmark_list, image_rows, image_cols):
    _VISIBILITY_THRESHOLD = 0.5
    _PRESENCE_THRESHOLD = 0.5
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                    image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates

# algorithm to recognize action
class CodeRecognitioner:

    def __init__(self) -> None:
        self.code = None
        self.is_raising = False
        self.time_raising = 0
        self.raise_time = 0
        self.pre_is_raise = False
        self.is_downing = False
        self.time_downing = 0


    def append(self, code):
        final_rslt = code[0]
        # 持续4次才算
        if code[0] and self.time_raising < 6:
            self.time_raising += 1
        elif code[0]:
            self.is_raising = True
        else:
            self.is_raising = False
            self.time_raising = 0

        # 上一期举手则触发 -> 这一段时间遇到举手/其他则不增加raise_time
        if self.is_downing and self.time_downing < 10:
            self.time_downing += 1
        else:
            self.is_downing = False
            self.time_downing = 0

        if self.is_raising and not self.pre_is_raise and (not self.is_downing):
            self.is_downing = True
            self.time_downing = 0
            self.raise_time += 1
        self.pre_is_raise = self.is_raising
        return self.is_raising, self.raise_time
        

class ColaPose:

    def __init__(self, is_visualize=False, video_path="0") -> None:
        self.cap = cv2.VideoCapture(0 if video_path=='0' else video_path)
        if not self.cap.isOpened():
            raise Exception("Video in \"{}\" open failed.".format(video_path))
        self.is_visualize = is_visualize
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.code_recog = CodeRecognitioner()

    def release_cap(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release_cap()

    def get_frame(self):
        return self.cap.read()

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
        theta1 = dot_product_angle(kp_16[0:2] - kp_14[0:2], np.asarray([1, 0], dtype=np.float32))  # (0, 50, 90) flat -> vertical
        theta2 = (rslt.pose_landmarks.landmark[16].visibility >= 0.5) and (rslt.pose_landmarks.landmark[14].visibility >= 0.5)
        return [theta1, theta2]


    def run_recognition(self, is_visualize=False):
        while self.cap.isOpened():
            ret, img = self.get_frame()
            if not ret:
                print("Camera read error.")
                break
            rslt = self.reference_pose(img)  # img -> output(of model)

            ##### Algorithem for recognition
            code = self.encode_pose_theta(rslt_processed)  # [True, xxx] -> OK
            rslt_processed = self.processe_pose_rslt(rslt)  # DONE:output(of model) -> feature

            # prnt node info
            image_rows, image_cols, _ = img.shape
            landmark_px = landmark_to_pixel(rslt.pose_landmarks, image_rows, image_cols)
            final_rslt = self.code_recog.append(code)
            # print(landmark_px[16], landmark_px[14])
            print("腕关节坐标: {}".format(landmark_px[16] if 16 in landmark_px.keys() else None), 
                  "肘关节关节坐标: {}" .format(landmark_px[14] if 14 in landmark_px.keys() else None),  
                  "处于举手状态" if final_rslt[0] else "未处于举手状态", 
                  "举手次数: {}".format(final_rslt[1]), 
                  sep='\t')

            ##### End of recognition
            if self.is_visualize:
                self.draw_pose_notation(img, rslt)
                rslt_processed = self.processe_pose_rslt(rslt)
                code = self.encode_pose_theta(rslt_processed)
                cv2.imshow("WIN", img)

            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def encode_pose_theta(self, theta_list):
        code_theta = [False] * 2
        code_theta[0] = theta_list[0]<=110 and theta_list[0] >= 50  # theta1 \in (50, 90)->True else False
        code_theta[0] = code_theta[0] and theta_list[1]
        return code_theta


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

def get_config():
    parse = argparse.ArgumentParser()
    parse.add_argument('-v', '--video-path', type=str, default="0", help="Path to video path, camera=0.")
    parse.add_argument('-is-visualize', action='store_true', help="Is show video.")
    return parse.parse_args()

if __name__ == "__main__":
    config = get_config()
    cola_pose = ColaPose(config.is_visualize, config.video_path)
    cola_pose.run_recognition()