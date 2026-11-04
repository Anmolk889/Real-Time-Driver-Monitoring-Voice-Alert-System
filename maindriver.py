import cv2
import mediapipe as mp
import math
import time
import threading
import winsound
import pyttsx3
import numpy as np
from fer.fer import FER
alarm_playing = False
voice_triggered = False

engine = pyttsx3.init()
engine.setProperty("rate", 170)

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        threading.Thread(
            target=lambda: winsound.PlaySound(
                "alarm.wav",
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
            ),
            daemon=True
        ).start()

def stop_alarm():
    global alarm_playing
    alarm_playing = False
    winsound.PlaySound(None, winsound.SND_PURGE)

def speak(text):
    threading.Thread(target=lambda: _speak_thread(text), daemon=True).start()

def _speak_thread(text):
    engine.say(text)
    engine.runAndWait()

def dist(a, b):
    return math.dist(a, b)

def calculate_ear(landmarks, eye):
    A = dist(landmarks[eye[1]], landmarks[eye[5]])
    B = dist(landmarks[eye[2]], landmarks[eye[4]])
    C = dist(landmarks[eye[0]], landmarks[eye[3]])
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks):
    vertical = dist(landmarks[13], landmarks[14])
    horizontal = dist(landmarks[78], landmarks[308])
    return vertical / horizontal

def get_head_pose(face_landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),
        (face_landmarks[152].x * w, face_landmarks[152].y * h),
        (face_landmarks[263].x * w, face_landmarks[263].y * h),
        (face_landmarks[33].x * w, face_landmarks[33].y * h),
        (face_landmarks[287].x * w, face_landmarks[287].y * h),
        (face_landmarks[57].x * w, face_landmarks[57].y * h)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),    
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw = angles[1]

    return pitch, yaw

def draw_panel(frame, ear, mar, pitch, yaw, status, yawn_count):
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (370, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "AI DRIVER MONITORING", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"EAR: {ear:.2f}", (25, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.putText(frame, f"MAR: {mar:.2f}", (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)

    cv2.putText(frame, f"Pitch: {pitch:.1f}", (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Yaw: {yaw:.1f}", (170, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Yawn Count: {yawn_count}", (25, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if status == "NORMAL":
        color = (0, 255, 0)
    elif status == "YAWNING":
        color = (0, 255, 255)
    elif status == "LOOKING AWAY":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (10, 190), (370, 240), color, -1)

    cv2.putText(frame, status, (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
MOUTH_THRESHOLD = 0.60

LOOK_AWAY_YAW_THRESHOLD = 40
LOOK_AWAY_PITCH_THRESHOLD = 25

sleep_start_time = None
yaw_start_time = None
look_away_start_time = None

yawn_count = 0
last_yawn_time = 0

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE"
    ear = 0
    mar = 0
    pitch = 0
    yaw = 0

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face,
                mp_face_mesh.FACEMESH_TESSELATION,
                None,
                mp_styles.get_default_face_mesh_tesselation_style()
            )

            landmarks = []

            for lm in face.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            mar = calculate_mar(landmarks)

            pitch, yaw = get_head_pose(face.landmark, frame.shape)

            status = "NORMAL"

            
            if mar > MOUTH_THRESHOLD:
                if yaw_start_time is None:
                    yaw_start_time = time.time()

                if time.time() - yaw_start_time > 1.5:
                    status = "YAWNING"

                    if time.time() - last_yawn_time > 5:
                        yawn_count += 1
                        last_yawn_time = time.time()
            else:
                yaw_start_time = None

            looking_away = abs(yaw) > LOOK_AWAY_YAW_THRESHOLD or abs(pitch) > LOOK_AWAY_PITCH_THRESHOLD

            if looking_away:
                if look_away_start_time is None:
                    look_away_start_time = time.time()

                if time.time() - look_away_start_time > 1.5:
                    status = "LOOKING AWAY"
            else:
                look_away_start_time = None


            if ear < EAR_THRESHOLD:

                if sleep_start_time is None:
                    sleep_start_time = time.time()

                sleep_duration = time.time() - sleep_start_time

                if sleep_duration >= 3 or (yawn_count >= 2 and sleep_duration >= 1):
                    status = "DROWSINESS ALERT"
                    play_alarm()

                    if not voice_triggered:
                        speak("Warning. Stop the car.")
                        voice_triggered = True
            else:
                sleep_start_time = None

                if status == "NORMAL":
                    stop_alarm()
                    voice_triggered = False

            if status == "LOOKING AWAY":
                play_alarm()

                if not voice_triggered:
                    speak("Please look forward.")
                    voice_triggered = True

            draw_panel(frame, ear, mar, pitch, yaw, status, yawn_count)

    else:
        stop_alarm()
        voice_triggered = False

    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
stop_alarm()
cv2.destroyAllWindows()