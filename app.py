#pip install flask numpy pillow tensorflow

from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import uuid

app = Flask(__name__)
model = load_model('best_pneumonia_model.h5')  

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')

    # Crop center square
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                    (w + min_dim) // 2, (h + min_dim) // 2))

    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            pred = model.predict(img_array)[0][0]
            label = 'PNEUMONIA' if pred > 0.5 else 'NORMAL'
            confidence = pred if pred > 0.5 else 1 - pred

            prediction = f"{label} ({confidence * 100:.2f}%)"
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
