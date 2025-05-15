import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

IMG_WIDTH = 540
IMG_HEIGHT = 420

def create_model():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

def process_image(image_path):
    img = cv2.imread(image_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (1, IMG_HEIGHT, IMG_WIDTH, 1))
    return img

model = create_model()
model.load_weights('my_checkpoint.weights.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Файл не найден', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'Файл не выбран', 400
        
        if file:
            filename = secure_filename(file.filename)
            base_name, ext = os.path.splitext(filename)
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            processed_img = process_image(input_path)
            result = model.predict(processed_img)
            
            output_filename = f'denoised_{base_name}.png'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            result_img = (result[0, :, :, 0] * 255).astype(np.uint8)
            cv2.imwrite(output_path, result_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            return render_template('result.html', 
                                 original_image=filename,
                                 processed_image=output_filename)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 