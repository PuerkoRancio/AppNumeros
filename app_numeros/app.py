from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import base64
import re
import os


app = Flask(__name__, static_folder='static', template_folder='templates')

# Ruta para servir archivos estáticos
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Cargar el modelo solo cuando se use (evita errores al iniciar)
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    with tf.keras.utils.custom_object_scope({'softmax_v2': tf.keras.activations.softmax}):
        return load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No se recibió imagen'}), 400
            
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        if not image_data:
            return jsonify({'error': 'Datos de imagen vacíos'}), 400
            
        # Decodificar y procesar la imagen
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convertir a escala de grises y redimensionar
        image = image.convert('L').resize((28, 28))
        
        # Convertir a array numpy
        image_array = np.array(image)
        
        # Normalización e inversión correcta
        image_array = image_array.astype('float32') / 255.0
        image_array = 1 - image_array  # Invertir (MNIST tiene fondo negro)
        
        # Verificar rango de valores
        print(f"Valores mínimo/máximo: {image_array.min()}, {image_array.max()}")
        
        # Reformatear para el modelo (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        model = get_model()
        prediction = model.predict(image_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        print(f"Predicción: {predicted_digit} (Confianza: {confidence:.2f})")
        print(f"Distribución completa: {prediction[0]}")
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        })
        
    except Exception as e:
        print(f"Error en /predict: {str(e)}")  # Log detallado
        return jsonify({'error': str(e)}), 500
    
@app.route('/model_summary', methods=['GET'])
def model_summary():
    try:
        model = get_model()
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        return jsonify({'summary': '\n'.join(summary)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))