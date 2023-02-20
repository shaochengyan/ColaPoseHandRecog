import os
import cv2
import numpy as np
from PIL import Image


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product /
                           (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0


def scale_img_with_cons_ration(img, win_w=800, win_h=600):
    """ scale img to fixed win size but with constant image size/ration
    img: (w, h, c)
    return: ndarray img (win_h, win_w, c)
    """
    h, w = img.shape[0:2]
    ratio_img = w / h
    if ratio_img > win_w / win_h:
        new_w = win_w
        new_h = int(new_w / ratio_img)
    else:
        new_h = win_h
        new_w = int(new_h * ratio_img)

    # resize
    img = cv2.resize(img, (new_w, new_h))

    # padding
    top_pad_num = round((win_h - new_h) / 2)
    bot_pad_num = win_h - new_h - top_pad_num
    left_pad_num = round((win_w - new_w) / 2)
    right_pad_num = win_w - new_w - left_pad_num
    img = cv2.copyMakeBorder(img,
                             top_pad_num,
                             bot_pad_num,
                             left_pad_num,
                             right_pad_num,
                             cv2.BORDER_CONSTANT,
                             value=0)
    return img