import numpy as np

def landmark_to_ndarray(landmark):
    return np.asarray([landmark.x, landmark.y, landmark.z])

def landmark_to_list(landmark, is_vis=False):
    if is_vis:
        return [landmark.x, landmark.y, landmark.z, landmark.visibility]
    else:
        return [landmark.x, landmark.y, landmark.z]
