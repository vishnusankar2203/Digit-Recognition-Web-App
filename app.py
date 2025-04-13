from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained model
model = load_model('mnist_cnn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(img_data)).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match MNIST input
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)[0]
    predicted_digit = int(np.argmax(prediction))
    confidence = [float(np.round(prob, 3)) for prob in prediction]

    return jsonify({
        'prediction': predicted_digit,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
