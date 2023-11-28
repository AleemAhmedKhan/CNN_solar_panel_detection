from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name)

# Load your VGG model (load the model as needed)
vgg_model = load_model('vgg_model.h5')

# Define an allowed file extension for the uploaded image
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(100, 100))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize

    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            # Securely save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Preprocess the uploaded image
            img = preprocess_image(file_path)

            # Get the prediction from the VGG model (modify this part accordingly)
            prediction = vgg_model.predict(img)

            # Assuming the model outputs class probabilities, you can take the class with the highest probability
            class_id = np.argmax(prediction)

            # Clean up by removing the uploaded image
            os.remove(file_path)

            return jsonify({'prediction': class_id})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
