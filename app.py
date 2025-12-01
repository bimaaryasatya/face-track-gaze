from flask import Flask, render_template, g, request
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64
import sqlite3
import time
import datetime


# -----------------------
# CONFIG
# -----------------------
DB_PATH = "cheat_logs.db"
HEAD_THRESHOLD = 20    # jumlah frame untuk dianggap menoleh
GAZE_THRESHOLD = 15    # jumlah frame untuk gaze kiri/kanan
# -----------------------

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=10, ping_timeout=30)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------
# SIMPLE SQLITE HELPERS
# -----------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        socket_id TEXT,
        event_type TEXT,
        detail TEXT,
        timestamp TEXT
    )
    """)
    db.commit()
    db.close()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def log_event(socket_id, event_type, detail=""):
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO events (socket_id, event_type, detail, timestamp) VALUES (?, ?, ?, ?)",
                (socket_id, event_type, detail, datetime.datetime.now(datetime.timezone.utc).isoformat()
))
    db.commit()

# Init DB on server start
init_db()

# -----------------------
# Per-socket counters storage
# -----------------------
# Structure: { sid: { 'head_counter': int, 'gaze_counter': int, 'last_seen': ts } }
clients = {}

# -----------------------
# PROCESS FRAME
# -----------------------
def analyze_face(frame):
    """
    Returns processed_frame, status_info dict:
    status_info = {
      'num_faces': int,
      'head_alert': bool,
      'gaze_dir': 'Kiri'|'Kanan'|'Tengah'|'Unknown'
    }
    """
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    status_info = {
        'num_faces': 0,
        'head_alert': False,
        'gaze_dir': 'Unknown'
    }

    status = "Normal"
    color = (0, 255, 0)

    if results.multi_face_landmarks:
        num_faces = len(results.multi_face_landmarks)
        status_info['num_faces'] = num_faces

        cv2.putText(frame, f"Wajah terdeteksi: {num_faces}",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        if num_faces > 1:
            status = "⚠️ Multiple Faces Detected"
            color = (0, 0, 255)
            cv2.putText(frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            return frame, status_info

        # single face
        face_landmarks = results.multi_face_landmarks[0]

        # HEAD ORIENTATION
        nose = face_landmarks.landmark[1]
        left_eye_outer = face_landmarks.landmark[33]
        right_eye_outer = face_landmarks.landmark[263]

        nose_x = int(nose.x * w)
        left_x = int(left_eye_outer.x * w)
        right_x = int(right_eye_outer.x * w)

        ratio_head = (right_x - nose_x) / (nose_x - left_x + 1e-6)
        if ratio_head < 0.6 or ratio_head > 1.6:
            status = "⚠️ Menoleh - Potensi Cheating"
            color = (0, 0, 255)
            status_info['head_alert'] = True

        # EYE GAZE
        try:
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            left_pupil = face_landmarks.landmark[468]

            right_eye_left = face_landmarks.landmark[362]
            right_eye_right = face_landmarks.landmark[263]
            right_pupil = face_landmarks.landmark[473]

            left_ratio = (left_pupil.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x + 1e-6)
            right_ratio = (right_pupil.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x + 1e-6)
            gaze_ratio = (left_ratio + right_ratio) / 2

            if gaze_ratio < 0.35:
                gaze_dir = "Kiri"
            elif gaze_ratio > 0.65:
                gaze_dir = "Kanan"
            else:
                gaze_dir = "Tengah"
            status_info['gaze_dir'] = gaze_dir

            # draw small circles for these landmarks
            for i in [33, 133, 362, 263, 468, 473]:
                lm = face_landmarks.landmark[i]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                           2, (0, 255, 255), -1)
            cv2.putText(frame, f"Gaze: {gaze_dir}",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
        except Exception:
            status_info['gaze_dir'] = 'Unknown'

    else:
        status = "Wajah tidak terdeteksi"
        color = (0, 165, 255)

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame, status_info

# -----------------------
# SOCKET.IO HANDLERS
# -----------------------
@socketio.on("connect")
def on_connect():
    sid = request.sid
    clients[sid] = {'head_counter': 0, 'gaze_counter': 0, 'last_seen': time.time()}
    print(f"[CONNECT] {sid}")
    log_event(sid, "connect", "client connected")
    emit("connected", {"message": "connected", "sid": sid})

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    print(f"[DISCONNECT] {sid}")
    log_event(sid, "disconnect", "client disconnected")
    clients.pop(sid, None)

@socketio.on("send_frame")
def receive_frame(base64_data):
    sid = request.sid
    clients.setdefault(sid, {'head_counter': 0, 'gaze_counter': 0, 'last_seen': time.time()})
    clients[sid]['last_seen'] = time.time()

    try:
        # decode frame
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        jpg = base64.b64decode(base64_data)
        np_arr = np.frombuffer(jpg, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # analyze
        processed_frame, info = analyze_face(frame)

        # update counters based on analysis
        head_alert = info.get('head_alert', False)
        gaze_dir = info.get('gaze_dir', 'Unknown')
        num_faces = info.get('num_faces', 0)

        # reset if no face
        if num_faces == 0:
            clients[sid]['head_counter'] = max(0, clients[sid]['head_counter'] - 1)
            clients[sid]['gaze_counter'] = max(0, clients[sid]['gaze_counter'] - 1)
        else:
            if head_alert:
                clients[sid]['head_counter'] += 1
            else:
                clients[sid]['head_counter'] = max(0, clients[sid]['head_counter'] - 1)

            if gaze_dir in ("Kiri", "Kanan"):
                clients[sid]['gaze_counter'] += 1
            else:
                clients[sid]['gaze_counter'] = max(0, clients[sid]['gaze_counter'] - 1)

        # If counters cross thresholds -> emit cheating_alert + log
        reasons = []
        if clients[sid]['head_counter'] > HEAD_THRESHOLD:
            reasons.append("Menoleh terlalu lama")
        if clients[sid]['gaze_counter'] > GAZE_THRESHOLD:
            reasons.append("Melihat keluar layar terlalu lama")
        if num_faces > 1:
            reasons.append("Multiple faces detected")

        if reasons:
            detail = "; ".join(reasons)
            print(f"[ALERT] {sid} -> {detail}")
            emit("cheating_alert", {"sid": sid, "detail": detail})
            log_event(sid, "cheating_alert", detail)
            # Optionally, reset counters or reduce so it won't spam
            clients[sid]['head_counter'] = 0
            clients[sid]['gaze_counter'] = 0

        # encode processed frame and send back
        _, buf = cv2.imencode(".jpg", processed_frame)
        processed_b64 = base64.b64encode(buf).decode("utf-8")
        emit("processed_frame", processed_b64)

    except Exception as e:
        print("ERROR processing frame:", e)
        log_event(sid, "error", str(e))

# -----------------------
# Logs viewer (simple)
# -----------------------
@app.route("/logs")
def view_logs():
    db = get_db()
    cur = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()
    html = "<h2>Cheating Logs (last 200)</h2><table border='1' cellpadding='6'><tr><th>id</th><th>socket</th><th>type</th><th>detail</th><th>timestamp (UTC)</th></tr>"
    for r in rows:
        html += f"<tr><td>{r['id']}</td><td>{r['socket_id']}</td><td>{r['event_type']}</td><td>{r['detail']}</td><td>{r['timestamp']}</td></tr>"
    html += "</table>"
    return html

# -----------------------
# Main
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
