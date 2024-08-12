from flask import Flask, request, render_template, send_from_directory
from pymongo import MongoClient
import os
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from model import straighten_license_plate, extract_text_from_image  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')
client = MongoClient("mongodb://mongodb:27017/", serverSelectionTimeoutMS=50000)

db = client['license_plate_db']
Licence_collection = db['LicencePlate']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Create a unique filename to avoid overwriting
        filename = 'temp_image'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = Image.open(filepath)
        image = image.resize((650, 350))  # Resize if needed
        straightened_image = straighten_license_plate(image)
        
        # Save only the straightened image
        straightened_filename = 'straightened_' + file.filename
        straightened_filepath = os.path.join(app.config['UPLOAD_FOLDER'], straightened_filename)
        straightened_image.save(straightened_filepath)
        
        # Extract text from the straightened image
        text = extract_text_from_image(straightened_image)
        
        # Clean up the temporary uploaded file
        os.remove(filepath)
        
        # Store data in MongoDB
        image_data = {
            'filename': straightened_filename,
            'image_path': straightened_filepath
        }
        text_data = {
            'text': text
        }
        datetime_data = {
            'datetime': datetime.now()
        }
        
        Licence_collection.insert_many([image_data,text_data,datetime_data])
   
        
        return render_template('index.html', 
                               straightened_image='uploads/' + straightened_filename,
                               extracted_text=text)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

