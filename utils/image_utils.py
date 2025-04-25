import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2

# Load Xception model
model = Xception(weights='imagenet', include_top=True)

def predict_deepfake_from_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    predicted_class = tf.keras.applications.xception.decode_predictions(preds, top=1)[0][0][1]

    return predicted_class
