from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os

app = Flask(__name__)

# -------------------------------
# LOAD KNOWN FACES
# -------------------------------
known_encodings = []
known_ids = []

for file in os.listdir("known_faces"):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        sid = os.path.splitext(file)[0]
        img = face_recognition.load_image_file("known_faces/" + file)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            known_encodings.append(enc[0])
            known_ids.append(sid)
            print("Loaded:", sid)


# -------------------------------
# FACE CHECK
# -------------------------------
@app.route("/check_face", methods=["POST"])
def check_face():
    file = request.files["image"]
    buf = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    encs = face_recognition.face_encodings(frame)
    if len(encs) == 0:
        return jsonify({"match": False})

    enc = encs[0]
    match = face_recognition.compare_faces(known_encodings, enc)

    if True in match:
        idx = match.index(True)
        sid = known_ids[idx]
        return jsonify({"match": True, "id": sid})

    return jsonify({"match": False})


# -------------------------------
# UNIFORM CHECK (COLOR BASED)
# -------------------------------
@app.route("/check_uniform", methods=["POST"])
def check_uniform():
    file = request.files["image"]
    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- expected uniform colors ---
    # white shirt
    white_low = np.array([0, 0, 170])
    white_high = np.array([179, 30, 255])

    # navy blue vest, tie, pant
    navy_low = np.array([90, 80, 20])
    navy_high = np.array([130, 255, 80])

    # black shoes
    black_low = np.array([0, 0, 0])
    black_high = np.array([179, 255, 40])

    mask_white = cv2.inRange(hsv, white_low, white_high)
    mask_navy = cv2.inRange(hsv, navy_low, navy_high)
    mask_black = cv2.inRange(hsv, black_low, black_high)

    w = cv2.countNonZero(mask_white)
    n = cv2.countNonZero(mask_navy)
    b = cv2.countNonZero(mask_black)

    uniform_ok = (w > 3000 and n > 3000 and b > 1500)

    return jsonify({
        "uniform_ok": uniform_ok,
        "white_detected": int(w),
        "navy_detected": int(n),
        "black_detected": int(b)
    })


@app.route("/")
def home():
    return render_template("index.html")


app.run(host="0.0.0.0", port=8000)
