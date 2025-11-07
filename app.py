from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DROIDCAM_URL = "http://192.168.0.200:4747/video"

def open_camera():
    cap = cv2.VideoCapture(DROIDCAM_URL)
    time.sleep(1)
    return cap

cap = open_camera()

def gen_frames():
    cheating_counter = 0
    gaze_counter = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("🔁 Reconnect ke DroidCam...")
            cap.open(DROIDCAM_URL)
            time.sleep(1)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        status = "Normal"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Titik kepala (nose + eyes)
            nose = face_landmarks.landmark[1]
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]

            nose_x = int(nose.x * w)
            left_x = int(left_eye_outer.x * w)
            right_x = int(right_eye_outer.x * w)

            # Rasio orientasi kepala
            ratio_head = (right_x - nose_x) / (nose_x - left_x + 1e-6)

            if ratio_head < 0.6 or ratio_head > 1.6:
                cheating_counter += 1
                status = "Menoleh - Potensi Cheating"
                color = (0, 0, 255)
            else:
                cheating_counter = max(0, cheating_counter - 1)

            # -----------------------------
            # Deteksi Arah Pandangan (Gaze)
            # -----------------------------
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            left_pupil = face_landmarks.landmark[468]

            right_eye_left = face_landmarks.landmark[362]
            right_eye_right = face_landmarks.landmark[263]
            right_pupil = face_landmarks.landmark[473]

            # Posisi relatif pupil kiri dan kanan
            left_ratio = (left_pupil.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x + 1e-6)
            right_ratio = (right_pupil.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x + 1e-6)
            gaze_ratio = (left_ratio + right_ratio) / 2

            # Gaze direction
            if gaze_ratio < 0.35:
                gaze_direction = "Kiri"
                gaze_counter += 1
            elif gaze_ratio > 0.65:
                gaze_direction = "Kanan"
                gaze_counter += 1
            else:
                gaze_direction = "Tengah"
                gaze_counter = max(0, gaze_counter - 1)

            # Tambahkan teks pandangan
            cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Jika menoleh atau gaze terlalu lama ke kiri/kanan
            if gaze_counter > 15 or cheating_counter > 20:
                status = "⚠️ CHEATING DETECTED"
                color = (0, 0, 255)

            # Gambar landmark mata
            for i in [33, 133, 362, 263, 468, 473]:
                lm = face_landmarks.landmark[i]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 255), -1)

        else:
            status = "Wajah tidak terdeteksi"
            color = (0, 165, 255)

        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
