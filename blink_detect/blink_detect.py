import cv2
import dlib
import numpy as np
import time
from flask import Flask, Response, render_template, request, jsonify
import os
import threading

app = Flask(__name__)

frame_width = 640
frame_height = 480

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

# Dlib의 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()
predictor_file = os.path.join(os.path.dirname(__file__), '/Users/gimseongmin/Desktop/ai경진대회/blink_detect/shape_predictor_68_face_landmarks (1).dat')

# 파일 경로 확인
if not os.path.exists(predictor_file):
    print(f'--(!)Error: {predictor_file} not found')
    exit(0)

predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25
closed_limit = 35
drowsiness_count = 0
show_frame = None
sign = None
color = (0, 255, 0)
capture = None
detecting = False

# 새로운 변수 추가
start_time = None
drowsy_duration = 3  # 3초

def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness():
    global number_closed
    global drowsiness_count
    global color
    global show_frame
    global sign
    global status
    global RIGHT_EYE
    global LEFT_EYE
    global EYES
    global capture
    global detecting
    global start_time

    while True:
        if capture is not None and detecting:
            ret, frame = capture.read()
            if not ret:
                print('Could not read frame')
                continue

            image = cv2.resize(frame, (frame_width, frame_height))
            show_frame = image
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_gray = clahe.apply(frame_gray)
            
            faces = detector(frame_gray)

            if len(faces) == 0:
                cv2.putText(show_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            for rect in faces:
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                points = predictor(image, rect)
                if points.num_parts != 68:
                    print(f'Error: Detected {points.num_parts} parts, but expected 68')
                    continue
                
                points = np.matrix([[p.x, p.y] for p in points.parts()])
                show_parts = points[EYES]

                if show_parts.shape[1] != 2:
                    print('Error: Unexpected number of values in show_parts')
                    continue

                right_eye_EAR = getEAR(points[RIGHT_EYE])
                left_eye_EAR = getEAR(points[LEFT_EYE])
                mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2

                right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
                left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

                cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0, 0], right_eye_center[0, 1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0, 0], left_eye_center[0, 1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if mean_eye_EAR < min_EAR:
                    if start_time is None:
                        start_time = time.time()
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= drowsy_duration:
                        drowsiness_count += 1
                        start_time = None
                        print(f"Drowsy count increased: {drowsiness_count}")
                    number_closed += 1
                    color = (0, 0, 255)
                    status = 'Drowsy'
                    sign = 'Drowsy'
                else:
                    start_time = None
                    number_closed = 0
                    status = 'Awake'
                    color = (0, 255, 0)
                    sign = 'Awake'

                cv2.putText(image, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for point in show_parts:
                    x, y = point[0, 0], point[0, 1]
                    cv2.circle(image, (x, y), 1, color, -1)

            show_frame = image

        if cv2.waitKey(10) == 27:
            break

    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html', drowsiness_count=drowsiness_count)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global capture
    global detecting
    if capture is None:
        capture = cv2.VideoCapture(0)
        time.sleep(2.0)
        if not capture.isOpened():
            return "Could not open video", 500
    detecting = True
    return '', 204

@app.route('/get_drowsiness_count', methods=['GET'])
def get_drowsiness_count():
    return jsonify(drowsiness_count=drowsiness_count)

def gen():
    while True:
        if show_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', show_frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=detect_drowsiness).start()
    print("Starting server at http://localhost:8060")
    app.run(host='localhost', port=8060, debug=True)
