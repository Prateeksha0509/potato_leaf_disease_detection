import tensorflow as tf
import numpy as np
import cv2

classes = ["Early Blight", "Late Blight", "Healthy"]
model = tf.keras.models.load_model("models/potato_model.h5")

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return classes[np.argmax(pred)]