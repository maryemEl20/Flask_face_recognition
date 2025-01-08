import base64
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2 as cv
from face_recognition import recognize_face  # Importer la fonction de reconnaissance

app = Flask(__name__)

# Filtre pour encoder les images en base64
@app.template_filter('b64encode')
def b64encode_filter(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Lire l'image envoyée
            img = np.frombuffer(file.read(), np.uint8)
            img = cv.imdecode(img, cv.IMREAD_COLOR)
            
            # Effectuer la reconnaissance faciale
            result_img = recognize_face(img)
            
            # Encoder l'image traitée en base64
            _, img_encoded = cv.imencode('.jpg', result_img)
            img_data = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            
            # Passer l'image encodée au template
            return render_template('index.html', img_data=img_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
