import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# cvzone expects the legacy `mediapipe.solutions` API. Newer MediaPipe
# builds keep solutions under `mediapipe.python.solutions`.
if not hasattr(mp, "solutions"):
    from mediapipe.python import solutions as _mp_solutions
    mp.solutions = _mp_solutions

from cvzone.HandTrackingModule import HandDetector
import math
import time
from collections import deque
import json
import os
from datetime import datetime


# ---------------- Setup ----------------
class SignLanguageTypingUI:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Load model
        self.model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)

        # Load labels
        self.labels = []
        with open("Model/labels.txt", "r") as f:
            for line in f:
                self.labels.append(line.strip().split(" ", 1)[1])

        # Parameters
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

        # State management
        self.stable_label = ""
        self.stable_conf = 0.0
        self.smoothed_preds = None
        self.current_word = ""
        self.letter_locked = False
        self.word_history = []
        self.sentence = ""

        # Performance tracking
        self.fps = 0
        self.fps_deque = deque(maxlen=30)
        self.prev_time = time.time()

        # Settings
        self.show_probabilities = True
        self.show_landmarks = True
        self.show_help = False
        self.auto_space = True
        self.save_to_file = True
        self.confidence_threshold = self.MIN_CONF

        # Colors
        self.colors = {
            'primary': (0, 150, 255),  # Orange
            'secondary': (0, 200, 100),  # Green
            'warning': (0, 50, 255),  # Red
            'success': (0, 255, 150),  # Teal
            'text': (255, 255, 255),  # White
            'bg_dark': (30, 30, 40),  # Dark gray
            'bg_light': (50, 50, 65),  # Light gray
            'highlight': (255, 100, 50),  # Coral
        }

        # UI dimensions
        self.ui_width = 400  # Increased width for all text
        self.panel_height = 720  # Full height

        # Create output directory
        if not os.path.exists("output"):
            os.makedirs("output")

    def draw_panel(self, frame):
        """Draw all text and controls in the right panel"""
        height, width = frame.shape[:2]

        # Semi-transparent panel background
        panel_bg = np.zeros((height, self.ui_width, 3), dtype=np.uint8)
        panel_bg[:] = self.colors['bg_dark']
        frame[0:height, width - self.ui_width:width] = cv2.addWeighted(
            frame[0:height, width - self.ui_width:width], 0.3,
            panel_bg, 0.7, 0
        )

        # Draw panel border
        cv2.line(frame, (width - self.ui_width, 0), (width - self.ui_width, height),
                 self.colors['primary'], 2)

        # Panel title
        cv2.putText(frame, "SIGN TYPE", (width - self.ui_width + 20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, self.colors['primary'], 2)

        # FPS display
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - self.ui_width + 20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)

        # Status indicator
        hand_status = "HAND DETECTED" if self.stable_label else "NO HAND"
        hand_color = self.colors['success'] if self.stable_label else self.colors['warning']
        cv2.putText(frame, f"Status: {hand_status}", (width - self.ui_width + 20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 1)

        # Confidence bar
        bar_x = width - self.ui_width + 20
        bar_y = 140
        bar_width = 300
        bar_height = 20

        # Draw background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.colors['bg_light'], -1)

        # Draw confidence level
        conf_width = int(bar_width * min(self.stable_conf, 1.0))
        bar_color = self.colors['success'] if self.stable_conf > self.AUTO_COMMIT_CONF else self.colors['primary']
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height),
                      bar_color, -1)

        cv2.putText(frame, f"Conf: {self.stable_conf:.1%}",
                    (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)

        # Current detection - Display the actual detected letter instead of "Left" or "Right"
        if self.stable_label:
            # Display the detected letter with confidence percentage
            detected_text = f"{self.stable_label}: {self.stable_conf:.1%}"
            cv2.putText(frame, detected_text,
                        (width - self.ui_width + 20, bar_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['highlight'], 2)

        # Current word display - REMOVED THE BOX
        word_y = bar_y + 80
        cv2.putText(frame, "CURRENT WORD:", (width - self.ui_width + 20, word_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 1)

        # Current word value with larger font
        if self.current_word:
            word_size = 1.2 + (len(self.current_word) / 50)
            cv2.putText(frame, self.current_word, (width - self.ui_width + 20, word_y + 40),
                        cv2.FONT_HERSHEY_DUPLEX, word_size, self.colors['highlight'], 2)
        else:
            cv2.putText(frame, "(Waiting for input...)", (width - self.ui_width + 20, word_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 1)

        # Sentence display - REMOVED THE BOX
        sentence_y = word_y + 90
        cv2.putText(frame, "SENTENCE:", (width - self.ui_width + 20, sentence_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 1)

        # Sentence text with wrapping
        sentence_text = self.sentence if self.sentence else "(Start typing...)"
        self.wrap_text(frame, sentence_text, (width - self.ui_width + 20, sentence_y + 30),
                       self.ui_width - 40, font_scale=0.7)

        # Controls section - No box here
        controls_y = sentence_y + 120
        cv2.putText(frame, "CONTROLS", (width - self.ui_width + 20, controls_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['primary'], 1)

        controls = [
            ("ENTER - Complete Word", controls_y + 30),
            ("BACKSPACE - Delete Char", controls_y + 50),
            ("SPACE - Add Space", controls_y + 70),
            ("C - Clear All", controls_y + 90),
            ("S - Save Sentence", controls_y + 110),
            ("H - Toggle Help", controls_y + 130),
            ("ESC - Exit", controls_y + 150)
        ]

        for text, y_pos in controls:
            cv2.putText(frame, text, (width - self.ui_width + 20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        # Settings indicators - No box here
        settings_y = controls_y + 180
        cv2.putText(frame, "SETTINGS", (width - self.ui_width + 20, settings_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['primary'], 1)

        settings = [
            (f"Auto-space: {'ON' if self.auto_space else 'OFF'}", settings_y + 25),
            (f"Save to file: {'ON' if self.save_to_file else 'OFF'}", settings_y + 45),
        ]

        for text, y_pos in settings:
            cv2.putText(frame, text, (width - self.ui_width + 20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

    def draw_main_display(self, frame, imgCrop, imgWhite):
        """Draw only video feed and processed images (no text)"""
        height, width = frame.shape[:2]

        # Draw processed images (crop and white background)
        if imgCrop is not None and imgCrop.size > 0:
            # Draw confidence indicator near the images
            if self.stable_label:
                conf_x = width - self.ui_width + 20  # Position it in the right panel
                conf_y = 250  # Adjusted position
                conf_text = f"{self.stable_label}: {self.stable_conf:.1%}"

                # Background for confidence text
                (text_width, text_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
                cv2.rectangle(frame,
                              (conf_x - 10, conf_y - text_height - 10),
                              (conf_x + text_width + 10, conf_y + 10),
                              self.colors['bg_dark'], -1)

                # Draw confidence text
                text_color = self.colors['success'] if self.stable_conf > self.AUTO_COMMIT_CONF else self.colors[
                    'warning']
                cv2.putText(frame, conf_text, (conf_x, conf_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, text_color, 3)

                # Draw auto-commit indicator
                if self.stable_conf >= self.AUTO_COMMIT_CONF and not self.letter_locked:
                    cv2.putText(frame, "AUTO-COMMIT", (conf_x, conf_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['success'], 2)

    def wrap_text(self, frame, text, position, max_width, font_scale=0.7, thickness=1):
        """Wrap text to fit within max_width"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = []
        words = text.split(' ')
        current_line = words[0]

        for word in words[1:]:
            test_line = current_line + ' ' + word
            (width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        line_height = int(30 * font_scale)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * line_height),
                        font, font_scale, self.colors['text'], thickness)

    def show_help_screen(self, frame):
        """Display help information"""
        height, width = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Help content
        help_lines = [
            "SIGN LANGUAGE TYPING SYSTEM - HELP",
            "",
            "HOW TO USE:",
            "1. Show hand signs to the camera",
            "2. System will detect letters automatically",
            "3. When confidence is high (>98%), letter is typed",
            "",
            "KEYBOARD CONTROLS:",
            "  ENTER      - Complete current word",
            "  SPACE      - Add space between words",
            "  BACKSPACE  - Delete last character",
            "  C          - Clear all text",
            "  S          - Save sentence to file",
            "  H          - Toggle this help screen",
            "  ESC        - Exit program",
            "",
            "INDICATORS:",
            "  Green confidence bar - Ready to type",
            "  AUTO-COMMIT text - Letter will be added",
            "  Red border - Low confidence detection",
            "",
            "Press H again to close help"
        ]

        # Draw help text
        y_start = height // 6
        for i, line in enumerate(help_lines):
            y_pos = y_start + i * 40
            color = self.colors['primary'] if i == 0 else self.colors['text']
            font_scale = 1.0 if i == 0 else 0.7
            cv2.putText(frame, line, (width // 6, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if i == 0 else 1)

        return frame

    def save_sentence(self):
        """Save current sentence to file with timestamp"""
        if not self.sentence.strip():
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/sentence_{timestamp}.txt"

        data = {
            "timestamp": timestamp,
            "sentence": self.sentence,
            "word_history": self.word_history,
            "settings": {
                "auto_space": self.auto_space,
                "confidence_threshold": self.confidence_threshold
            }
        }

        # Save as JSON for structured data
        with open(f"output/sentence_{timestamp}.json", "w") as f:
            json.dump(data, f, indent=2)

        # Also save as plain text
        with open(filename, "w") as f:
            f.write(self.sentence)

        print(f"Sentence saved to {filename}")
        return True

    def calculate_fps(self):
        """Calculate frames per second"""
        current_time = time.time()
        self.fps_deque.append(1 / (current_time - self.prev_time))
        self.fps = sum(self.fps_deque) / len(self.fps_deque)
        self.prev_time = current_time

    def process_frame(self, frame):
        """Process a single frame for hand detection"""
        hands, img = self.detector.findHands(frame, draw=self.show_landmarks)

        imgCrop = None
        imgWhite = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255

            # Clamp crop coordinates
            y1 = max(0, y - self.offset)
            y2 = min(frame.shape[0], y + h + self.offset)
            x1 = max(0, x - self.offset)
            x2 = min(frame.shape[1], x + w + self.offset)
            imgCrop = frame[y1:y2, x1:x2]

            if imgCrop.size != 0:
                crop_h, crop_w, _ = imgCrop.shape
                aspectRatio = crop_h / crop_w

                # Resize and center on white background
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

                # -------- Model Prediction --------
                imgInput = cv2.resize(imgWhite, (self.in_w, self.in_h))
                imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
                imgInput = imgInput.astype(np.float32)
                imgInput = (imgInput / 127.5) - 1.0
                imgInput = np.expand_dims(imgInput, axis=0)

                preds = self.model.predict(imgInput, verbose=0)[0]

                # Smoothing
                if self.smoothed_preds is None:
                    self.smoothed_preds = preds
                else:
                    self.smoothed_preds = (
                            self.SMOOTHING * self.smoothed_preds +
                            (1 - self.SMOOTHING) * preds
                    )

                # Top-2 confidence gap
                top2 = np.argsort(self.smoothed_preds)[-2:]
                best = top2[-1]
                second = top2[-2]

                best_conf = self.smoothed_preds[best]
                gap = best_conf - self.smoothed_preds[second]

                if best_conf > self.confidence_threshold and gap > self.MIN_GAP:
                    self.stable_label = self.labels[best]
                    self.stable_conf = best_conf

                # Auto-commit logic
                if self.stable_conf >= self.AUTO_COMMIT_CONF and not self.letter_locked:
                    self.current_word += self.stable_label
                    self.letter_locked = True
                    print(f"Typing: {self.current_word}")

                # Release lock when confidence drops
                if self.stable_conf < self.RELEASE_CONF:
                    self.letter_locked = False

        return imgCrop, imgWhite

    def run(self):
        """Main application loop"""
        print("\n" + "=" * 60)
        print("SIGN LANGUAGE TYPING SYSTEM")
        print("=" * 60)
        print("Starting... Press H for help, ESC to exit")

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Calculate FPS
            self.calculate_fps()

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Process frame for hand detection
            imgCrop, imgWhite = self.process_frame(frame)

            # Draw UI elements - ALL TEXT IS NOW IN THE RIGHT PANEL
            self.draw_main_display(frame, imgCrop, imgWhite)
            self.draw_panel(frame)  # This contains ALL text

            # Show help if enabled
            if self.show_help:
                frame = self.show_help_screen(frame)

            # Display the frame
            cv2.imshow("Sign Language Typing System", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER
                if self.current_word:
                    if self.auto_space and self.sentence and not self.sentence.endswith(' '):
                        self.sentence += ' '
                    self.sentence += self.current_word
                    self.word_history.append(self.current_word)
                    print(f"\nWord added: '{self.current_word}'")
                    print(f"Sentence: {self.sentence}")
                    self.current_word = ""
                    self.letter_locked = False

            elif key == 32:  # SPACE
                if self.sentence and not self.sentence.endswith(' '):
                    self.sentence += ' '
                    print("Space added")

            elif key == 8:  # BACKSPACE
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                    print(f"Typing: {self.current_word}")
                elif self.sentence:
                    self.sentence = self.sentence[:-1]
                    print(f"Sentence: {self.sentence}")

            elif key == ord('c'):  # CLEAR ALL
                self.current_word = ""
                self.sentence = ""
                self.word_history = []
                self.letter_locked = False
                print("All text cleared")

            elif key == ord('s'):  # SAVE
                if self.save_sentence():
                    cv2.putText(frame, "SAVED!", (50, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 2.0, self.colors['success'], 3)
                    print("Sentence saved successfully!")

            elif key == ord('h'):  # HELP
                self.show_help = not self.show_help
                print(f"Help screen {'shown' if self.show_help else 'hidden'}")

            elif key == ord('+'):  # Increase confidence threshold
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")

            elif key == ord('-'):  # Decrease confidence threshold
                self.confidence_threshold = max(0.3, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")

            elif key == 27:  # ESC
                print("\nExiting...")
                if self.sentence and self.save_to_file:
                    self.save_sentence()
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


# ---------------- Main Entry Point ----------------
if __name__ == "__main__":
    app = SignLanguageTypingUI()
    app.run()