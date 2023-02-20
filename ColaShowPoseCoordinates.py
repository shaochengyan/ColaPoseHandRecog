import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rslt_to_coordinates(rslt):
    if rslt.pose_landmarks is None:
        return None
    data_list = []
    for landmark in rslt.pose_world_landmarks.landmark:
        data_list.append([landmark.x, landmark.y, landmark.z])
    return np.asarray(data_list)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_title("3d Scatter plot")


plt.ion()
cap = cv2.VideoCapture(1)
ax.set_zlim3d(0, 2)  # viewrange for z-axis should be [-4,4]
ax.set_ylim3d(0, 3)  # viewrange for y-axis should be [-2,2]
ax.set_xlim3d(0, 3)  # viewrange for x-axis should be [-2,2]
# plt.zlim([0, 1])
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Image reading error.")
        break
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rslt = pose.process(img)

    data_coordinates = rslt_to_coordinates(rslt)
    if data_coordinates is None:
        continue
    # draw
    plt.cla()
    ax.scatter3D(data_coordinates[:, 0],
                 data_coordinates[:, 1],
                 data_coordinates[:, 2],
                 lw=2,
                 color=[0.8, 0.33, 0])
    ax.scatter3D(0, 0, 0, lw=2, color=[0, 1.0, 0.0])

    # draw line
    for connection in mp_pose.POSE_CONNECTIONS:
        idx0 = list(connection)
        ax.plot(data_coordinates[idx0, 0], data_coordinates[idx0, 1],
                data_coordinates[idx0, 2])
    plt.pause(0.2)

plt.ioff()
plt.show()