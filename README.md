# chest-xray-analysis-using-deep-learning

This project is a Flask-based web application designed to detect various chest-related diseases using deep learning models trained on chest X-ray images. Users can upload chest X-rays, and the app will classify the image into one of several diseases or mark it as healthy.

The dataset was sourced from [Kaggle, NIH Chest X-ray Dataset].

Features
Disease Detection: Supports detection of multiple diseases, such as:
Cardiomegaly
Mass
Nodule
Pneumonia
Pneumothorax
Effusion
Infiltration
Atelectasis


User-Friendly Interface: Simple interface for uploading images and viewing results.
JSON Response: Provides a JSON response with the disease prediction.
Technologies Used
Python: Core language for development
Flask: Web framework
TensorFlow / Keras: Deep learning library for training and loading models
Pillow (PIL): Image processing
HTML & CSS: For the web interface
