import os
import cv2
import time
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 감정 모델 경로 설정
file_path = '/Users/gimseongmin/Desktop/ai경진대회/emotion_detect/emotion_model.h5'
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad']

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 디버그 모드 활성화
app.config["DEBUG"] = True

# 파일 경로 확인 및 모델 로드
if os.path.exists(file_path):
    print("File exists.")
    try:
        emotion_model = load_model(file_path)
        print("Model loaded successfully.")
    except ValueError as e:
        print(f"Error loading model: {e}")
else:
    print(f"File does not exist at path: {file_path}")
    emotion_model = None

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image(image):
    # 히스토그램 균일화
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    # 감마 보정
    image = adjust_gamma(image, gamma=1.5)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if emotion_model is None:
        return jsonify({"error": "Emotion model is not loaded."})
    
    cap = cv2.VideoCapture(0)
    last_emotion = None
    emotion_start_time = time.time()
    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 이미지 전처리 적용
        processed_frame = preprocess_image(frame)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_locations = face_cascade.detectMultiScale(processed_frame, scaleFactor=1.3, minNeighbors=5)

        current_emotion = None

        for (x, y, w, h) in face_locations:
            face = processed_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            emotion_prediction = emotion_model.predict(face)[0]
            predicted_emotion_index = emotion_prediction.argmax()
            if predicted_emotion_index < len(emotion_labels):
                current_emotion = emotion_labels[predicted_emotion_index]
            else:
                current_emotion = None

            if current_emotion in emotion_labels:
                if current_emotion == last_emotion:
                    if time.time() - emotion_start_time >= 1.5:
                        detected_emotion = current_emotion
                        break
                else:
                    last_emotion = current_emotion
                    emotion_start_time = time.time()

        if detected_emotion:
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_emotion:
        return jsonify({"emotion": detected_emotion})
    else:
        return jsonify({"emotion": "No emotion detected"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070)
ㄴ