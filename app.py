import os
import uuid
import flask
import urllib
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
from flask import Flask, render_template, request

# Initialize Flask App
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = "E:/6th_project/Skin_Disease_Classification/skin_disease_classification_model.h5"
model = load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Class labels
classes = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'unknown',
    'vascular lesion',
    'warts molluscum'
]

def predict(filename, model):
    try:
        img = load_img(filename, target_size=(240, 240))
        img = img_to_array(img)
        img = img.reshape(1, 240, 240, 3)
        img = img.astype('float32') / 255.0

        result = model.predict(img)[0]

        # Identify top 4 predictions
        top_indices = np.argsort(result)[::-1][:4]
        class_result = [classes[i] for i in top_indices]
        prob_result = [(result[i] * 100).round(2) for i in top_indices]

        # Apply threshold: if top confidence is < 40%, classify as "unknown"
        threshold = 40.0
        if prob_result[0] < threshold:
            return ["unknown"], [prob_result[0]]

        return class_result, prob_result

    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["error"], [0.0]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success', methods=['POST'])
def success():
    target_img = os.path.join(os.getcwd(), 'static/images')

    if request.method == 'POST':
        error = ''
        
        if request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file_path = os.path.join(target_img, file.filename)
                file.save(file_path)

                class_result, prob_result = predict(file_path, model)

                if class_result[0] == "error":
                    error = "An error occurred during prediction. Please try again."
                    return render_template('index.html', error=error)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1] if len(class_result) > 1 else "N/A",
                    "class3": class_result[2] if len(class_result) > 2 else "N/A",
                    "class4": class_result[3] if len(class_result) > 3 else "N/A",
                    "prob1": prob_result[0],
                    "prob2": prob_result[1] if len(prob_result) > 1 else 0.0,
                    "prob3": prob_result[2] if len(prob_result) > 2 else 0.0,
                    "prob4": prob_result[3] if len(prob_result) > 3 else 0.0
                }
                return render_template('success.html', img=file.filename, predictions=predictions)

            else:
                error = "Please upload images in jpg, jpeg, or png format."

        return render_template('index.html', error=error)

if __name__ == "__main__":
    app.run(debug=True, port=4000)
