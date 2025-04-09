from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import re

app = Flask(__name__)

@app.route('/')
def home():
    return "¡Funciona en Render!"

# Cargar el modelo al iniciar la aplicación
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen en base64 del canvas
    image_data = request.form['image']
    
    # Eliminar el encabezado de la URL de datos
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    
    # Decodificar la imagen base64
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocesamiento de la imagen
    image = image.resize((28, 28)).convert('L')  # Redimensionar y convertir a escala de grises
    image_array = np.array(image)
    
    # Invertir colores (el modelo espera dígitos blancos sobre fondo negro)
    image_array = 255 - image_array
    
    # Normalizar y cambiar formato para el modelo
    image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Hacer la predicción
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)