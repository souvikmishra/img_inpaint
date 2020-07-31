import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras
from .pconv import PConv2D


def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

def imginp(masked_image, mask):
    pred =  keras.models.load_model(os.path.join(os.path.dirname(__file__),
                                                            "final_trained_model_test_main.h5"),
                                                            custom_objects={'dice_coef':dice_coef, 
                                                                        'PConv2D':PConv2D})
    # get images from path
    mi = cv2.imread(masked_image, -1)
    m = cv2.imread(mask, -1)
    # extract the alpha channel -- Need to add a check for those images which 
    # either don't have an alpha channel OR don't have the data in alpha channel
    alpha_mi, alpha_m = mi[::3], m[::3]
    new_mi = cv2.cvtColor(alpha_mi, cv2.COLOR_BGRA2BGR)
    new_m = cv2.cvtColor(alpha_m, cv2.COLOR_BGRA2BGR)
    # resize images to 32x32
    mi_r = cv2.resize(new_mi, (32,32))
    m_r = cv2.resize(new_m, (32,32))

    inp = [mi_r.reshape((1,) + mi_r.shape), m_r.reshape((1,) + m_r.shape)]
    val=pred.predict(inp)
    pred_img = cv2.resize(val.reshape(val.shape[1:]),(200,200))
    return cv2.imwrite("static/output/image.png", pred_img)



if __name__ == "__main__":
    imginp("/Users/retro/Desktop/img_inpaint/modelRunner/masked_image.png",
                    "/Users/retro/Desktop/img_inpaint/modelRunner/mask.png")
