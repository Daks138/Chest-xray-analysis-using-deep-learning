from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the saved models for each disease
model_cardiomegaly = load_model('Models/Cardiomegaly.h5')
model_mass = load_model('Models/Mass.h5')
model_nodule = load_model('Models/Nodule.h5')
model_pneumonia = load_model('Models/Pneumonia.h5')
model_pneumothorax = load_model('Models/Pneumothorax.h5')
model_effusion = load_model('Models/Effusion.h5')
model_infiltration = load_model('Models/Infiltration.h5')
model_atelectasis = load_model('Models/Atelectasis.h5')

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.ANTIALIAS)
    img_array = np.array(img)
    return img_array

# Function to predict disease for a single image
def predict_disease(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = preprocessed_image / 255.0  # Normalize pixel values
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    return prediction[0][0]

# Define routes for each page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('upload_prediction.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Define routes for each disease
@app.route('/predict/cardiomegaly', methods=['POST'])
def predict_cardiomegaly():
    return predict_disease_for('cardiomegaly', model_cardiomegaly)

# Define routes for each disease
@app.route('/predict/mass', methods=['POST'])
def predict_mass():
    return predict_disease_for('mass', model_mass)

@app.route('/predict/nodule', methods=['POST'])
def predict_nodule():
    return predict_disease_for('nodule', model_nodule)

@app.route('/predict/pneumonia', methods=['POST'])
def predict_pneumonia():
    return predict_disease_for('pneumonia', model_pneumonia)

@app.route('/predict/pneumothorax', methods=['POST'])
def predict_pneumothorax():
    return predict_disease_for('pneumothorax', model_pneumothorax)

@app.route('/predict/effusion', methods=['POST'])
def predict_effusion():
    return predict_disease_for('effusion', model_effusion)

@app.route('/predict/infiltration', methods=['POST'])
def predict_infiltration():
    return predict_disease_for('infiltration', model_infiltration)

@app.route('/predict/atelectasis', methods=['POST'])
def predict_atelectasis():
    return predict_disease_for('atelectasis', model_atelectasis)

# Function to handle prediction for a given disease
def predict_disease_for(disease, model):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_disease(filepath, model)

        # Get the specific disease name based on prediction probability
        disease_name = 'Positive' if prediction < 0.5 else 'Negative'

        result = {
            'prediction': disease_name,
            'image_path': f'/static/{filename}'
        }

        return jsonify(result)

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
