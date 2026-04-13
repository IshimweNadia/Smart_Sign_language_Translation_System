import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import tensorflow as tf

# ---------------- Setup ----------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)

labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        labels.append(line.strip().split(" ", 1)[1])

offset = 20
imgSize = 300
in_h = model.input_shape[1]
in_w = model.input_shape[2]

# ---------------- Decision Parameters ----------------
MIN_CONF = 0.55
MIN_GAP = 0.18
SMOOTHING = 0.7

AUTO_COMMIT_CONF = 0.98   # 🔑 commit only when VERY sure
RELEASE_CONF = 0.85       # must drop below this to allow next commit

stable_label = ""
stable_conf = 0.0
smoothed_preds = None

# ---------------- WORD TYPING ----------------
current_word = ""
letter_locked = False   # prevents repeated commits

# ---------------- Main Loop ----------------
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Clamp crop
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            crop_h, crop_w, _ = imgCrop.shape
            aspectRatio = crop_h / crop_w

            if aspectRatio > 1:
                k = imgSize / crop_h
                wCal = math.ceil(k * crop_w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / crop_w
                hCal = math.ceil(k * crop_h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # -------- Preprocessing --------
            imgInput = cv2.resize(imgWhite, (in_w, in_h))
            imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
            imgInput = imgInput.astype(np.float32)
            imgInput = (imgInput / 127.5) - 1.0
            imgInput = np.expand_dims(imgInput, axis=0)

            preds = model.predict(imgInput, verbose=0)[0]

            # -------- Smoothing --------
            if smoothed_preds is None:
                smoothed_preds = preds
            else:
                smoothed_preds = (
                    SMOOTHING * smoothed_preds +
                    (1 - SMOOTHING) * preds
                )

            # -------- Top-2 gap --------
            top2 = np.argsort(smoothed_preds)[-2:]
            best = top2[-1]
            second = top2[-2]

            best_conf = smoothed_preds[best]
            gap = best_conf - smoothed_preds[second]

            if best_conf > MIN_CONF and gap > MIN_GAP:
                stable_label = labels[best]
                stable_conf = best_conf

            # -------- AUTO COMMIT LOGIC --------
            if stable_conf >= AUTO_COMMIT_CONF and not letter_locked:
                current_word += stable_label
                letter_locked = True
                print("Typing:", current_word)

            # Release lock when confidence drops
            if stable_conf < RELEASE_CONF:
                letter_locked = False

            # -------- Draw --------
            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - 50),
                (x - offset + 160, y - offset),
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.putText(
                imgOutput,
                f"{stable_label} {stable_conf:.2f}",
                (x - offset + 5, y - offset - 15),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                imgOutput,
                f"WORD: {current_word}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset),
                (x + w + offset, y + h + offset),
                (255, 0, 255),
                4
            )

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER → submit word
        if current_word:
            print("\nFINAL WORD:", current_word)
            current_word = ""
            letter_locked = False

    elif key == 8:  # BACKSPACE
        current_word = current_word[:-1]
        print("Typing:", current_word)

    elif key == 27:  # ESC
        break
