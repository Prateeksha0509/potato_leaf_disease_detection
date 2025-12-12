from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
import os
import tensorflow as tf

model_path = os.path.join(os.getcwd(), "models", "potato_model.h5")
model = tf.keras.models.load_model(model_path)


classes = ["Early Blight", "Late Blight", "Healthy"]

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return classes[np.argmax(pred)]

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        file=request.files["image"]
        filepath="static/"+file.filename
        file.save(filepath)
        result=predict_image(filepath)
        return render_template("index.html", prediction=result, image_path=filepath)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)