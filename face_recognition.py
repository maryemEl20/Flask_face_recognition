import cv2 as cv
import numpy as np
import os

# Charger le classificateur Haar Cascade
haar_cascade = cv.CascadeClassifier('./data/Haarcascades/haar_face.xml')

# Liste des personnes à reconnaître
people = ['Cristiano Ronaldo', 'Lionel Messi']

# Charger ou entraîner le modèle de reconnaissance faciale
recognizer = cv.face.LBPHFaceRecognizer_create()

if os.path.exists('./face_recognizer.yml'):
    recognizer.read('./face_recognizer.yml')
else:
    faces, labels = [], []
    for person in people:
        folder = f'images/{person}'
        if not os.path.exists(folder):
            continue
        for image_name in os.listdir(folder):
            img = cv.imread(os.path.join(folder, image_name))
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in face_rects:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(people.index(person))
    recognizer.train(faces, np.array(labels))
    recognizer.save('./face_recognizer.yml')

# Fonction de reconnaissance faciale sur une image de test
def recognize_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in face_rects:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        
        if confidence < 40:
            name = people[label]
            color = (0, 255, 0)
        else:
            name = 'Inconnu'
            color = (0, 0, 255)
        
        # Dessiner le rectangle plus épais
        thickness = 3  # Épaisseur du rectangle
        cv.rectangle(img, (x, y), (x+w, y+h), color, thickness)
        
        # Ajouter un fond semi-transparent pour mieux afficher le texte
        overlay = img.copy()
        cv.rectangle(overlay, (x, y - 25), (x + w, y), color, -1)
        alpha = 0.5  # Transparence
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Afficher le texte avec une meilleure lisibilité
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv.getTextSize(name, font, font_scale, font_thickness)[0]
        text_x = x + (w - text_size[0]) // 2  # Centrer le texte horizontalement
        text_y = y - 5
        
        cv.putText(img, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    return img
