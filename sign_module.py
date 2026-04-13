import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
from cvzone.HandTrackingModule import HandDetector


class SignRecognition:
    def __init__(self, model_path="Model/keras_model.h5", labels_path="Model/labels.txt"):
        # Hand detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Load model
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Load labels
        self.labels = []
        with open(labels_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) > 1:
                    self.labels.append(parts[1])
                else:
                    self.labels.append(parts[0])

        # Image settings
        self.offset = 20
        self.imgSize = 300
        self.in_h = self.model.input_shape[1]
        self.in_w = self.model.input_shape[2]

        # Decision Parameters
        self.MIN_CONF = 0.55
        self.MIN_GAP = 0.18
        self.SMOOTHING = 0.7
        self.AUTO_COMMIT_CONF = 0.98
        self.RELEASE_CONF = 0.85

        self.stable_label = ""
        self.stable_conf = 0.0
        self.smoothed_preds = None
        self.letter_locked = False

        # UI: updated each process_frame()
        self.hand_detected = False
        self.display_conf = 0.0  # 0–1, top-class or stable confidence for the bar

        self.cap = None

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap.isOpened()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_frame(self, frame):
        hands, img = self.detector.findHands(frame, draw=True)
        new_char = None
        self.hand_detected = False
        self.display_conf = 0.0

        if hands:
            self.hand_detected = True
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255

            y1 = max(0, y - self.offset)
            y2 = min(frame.shape[0], y + h + self.offset)
            x1 = max(0, x - self.offset)
            x2 = min(frame.shape[1], x + w + self.offset)

            imgCrop = frame[y1:y2, x1:x2]

            if imgCrop.size != 0:
                crop_h, crop_w, _ = imgCrop.shape
                aspectRatio = crop_h / crop_w

                if aspectRatio > 1:
                    k = self.imgSize / crop_h
                    wCal = math.ceil(k * crop_w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    wGap = (self.imgSize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = self.imgSize / crop_w
                    hCal = math.ceil(k * crop_h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    hGap = (self.imgSize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Prepare input for model
                imgInput = cv2.resize(imgWhite, (self.in_w, self.in_h))
                imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
                imgInput = imgInput.astype(np.float32)
                imgInput = (imgInput / 127.5) - 1.0
                imgInput = np.expand_dims(imgInput, axis=0)

                preds = self.model.predict(imgInput, verbose=0)[0]

                # Smoothing predictions
                if self.smoothed_preds is None:
                    self.smoothed_preds = preds
                else:
                    self.smoothed_preds = (
                        self.SMOOTHING * self.smoothed_preds +
                        (1 - self.SMOOTHING) * preds
                    )

                # Get best predictions
                top2 = np.argsort(self.smoothed_preds)[-2:]
                best = top2[-1]
                second = top2[-2]

                best_conf = self.smoothed_preds[best]
                gap = best_conf - self.smoothed_preds[second]

                # Stability check
                if best_conf > self.MIN_CONF and gap > self.MIN_GAP:
                    self.stable_label = self.labels[best]
                    self.stable_conf = best_conf

                # Confidence for UI (prefer stable class when set)
                self.display_conf = float(self.stable_conf) if self.stable_label else float(best_conf)

                # Auto commit letter
                if self.stable_conf >= self.AUTO_COMMIT_CONF and not self.letter_locked:
                    new_char = self.stable_label
                    self.letter_locked = True

                # Release lock
                if self.stable_conf < self.RELEASE_CONF:
                    self.letter_locked = False
        else:
            self.stable_label = ""
            self.stable_conf = 0.0

        return frame, new_char