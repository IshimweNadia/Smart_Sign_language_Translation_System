import os
import json
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime

from sign_module import SignRecognition
from speech_module import SpeechRecognition


class CommunicationApp:
    AUTO_COMMIT_CONF = 0.98

    def __init__(self):
        self.mode = 0
        self.mode_names = ["SIGN LANGUAGE", "SPEECH RECOGNITION"]

        self.sign_recognizer = SignRecognition()
        self.speech_recognizer = SpeechRecognition(model_path="vosk-model")

        self.current_word = ""
        self.sentence = ""
        self.word_history = []
        self.running = True

        self.fps = 0.0
        self.fps_deque = deque(maxlen=30)
        self.prev_time = time.time()

        self.show_help = False
        self.auto_space = True
        self.save_to_file = True

        self.ui_width = 400

        self.colors = {
            "primary": (0, 150, 255),
            "secondary": (0, 200, 100),
            "warning": (0, 50, 255),
            "success": (0, 255, 150),
            "text": (255, 255, 255),
            "bg_dark": (30, 30, 40),
            "bg_light": (50, 50, 65),
            "highlight": (255, 100, 50),
            "word_blue": (255, 180, 0),
        }

        if not os.path.exists("output"):
            os.makedirs("output")

        self.speech_feedback_msg = ""
        self.speech_feedback_until = 0.0

        self.start_mode(self.mode)

    def start_mode(self, mode):
        if mode == 0:
            self.speech_recognizer.stop_listening()
            self.sign_recognizer.start_camera()
        else:
            self.sign_recognizer.stop_camera()
            self.speech_recognizer.start_listening()
        self.mode = mode

    def toggle_mode(self):
        self.start_mode(1 - self.mode)

    def calculate_fps(self):
        t = time.time()
        dt = t - self.prev_time
        if dt > 0:
            self.fps_deque.append(1.0 / dt)
        self.fps = sum(self.fps_deque) / len(self.fps_deque) if self.fps_deque else 0.0
        self.prev_time = t

    def wrap_text(self, frame, text, position, max_width, font_scale=0.7, thickness=1, line_limit=8):
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split()
        if not words:
            return
        current_line = words[0]
        lines = []
        for word in words[1:]:
            test_line = current_line + " " + word
            (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if tw <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        line_height = int(28 * font_scale)
        for i, line in enumerate(lines[:line_limit]):
            cv2.putText(frame, line, (x, y + i * line_height), font, font_scale, self.colors["text"], thickness)

    def draw_sign_panel(self, frame):
        h, w = frame.shape[:2]
        uw = self.ui_width

        panel = np.zeros((h, uw, 3), dtype=np.uint8)
        panel[:] = self.colors["bg_dark"]
        frame[0:h, w - uw : w] = cv2.addWeighted(frame[0:h, w - uw : w], 0.3, panel, 0.7, 0)
        cv2.line(frame, (w - uw, 0), (w - uw, h), self.colors["primary"], 2)

        x0 = w - uw + 20
        sr = self.sign_recognizer
        conf = float(sr.display_conf)

        cv2.putText(frame, "SIGN TYPE", (x0, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, self.colors["primary"], 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (x0, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors["text"], 1)

        if sr.hand_detected:
            status = "HAND DETECTED"
            st_color = self.colors["success"]
        else:
            status = "NO HAND"
            st_color = self.colors["warning"]
        cv2.putText(frame, f"Status: {status}", (x0, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, st_color, 1)

        bar_x, bar_y = x0, 128
        bar_w, bar_h = min(300, uw - 40), 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.colors["bg_light"], -1)
        fill = int(bar_w * min(1.0, max(0.0, conf)))
        bar_color = self.colors["success"] if conf >= self.AUTO_COMMIT_CONF else self.colors["primary"]
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
        pct = conf * 100.0
        cv2.putText(frame, f"Conf: {pct:.1f}%", (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colors["text"], 1)
        cv2.putText(frame, f"O: {pct:.1f}%", (bar_x + bar_w - 95, bar_y + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors["primary"], 1)

        if sr.stable_label:
            det = f"{sr.stable_label}: {sr.stable_conf:.1%}"
            cv2.putText(frame, det, (x0, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors["highlight"], 1)

        word_y = bar_y + 70
        cv2.putText(frame, "CURRENT WORD:", (x0, word_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.colors["text"], 1)
        if self.current_word:
            cv2.putText(frame, self.current_word, (x0, word_y + 36), cv2.FONT_HERSHEY_DUPLEX, 1.15, self.colors["word_blue"], 2)
        else:
            cv2.putText(frame, "(Waiting for input...)", (x0, word_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)

        sentence_y = word_y + 88
        cv2.putText(frame, "SENTENCE:", (x0, sentence_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.colors["text"], 1)
        sent = self.sentence if self.sentence else "(Start typing...)"
        self.wrap_text(frame, sent, (x0, sentence_y + 28), uw - 40, font_scale=0.62)

        controls_y = min(sentence_y + 120, h - 200)
        cv2.putText(frame, "CONTROLS", (x0, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors["primary"], 1)
        lines = [
            "ENTER: Complete Word",
            "BACKSPACE: Delete Char",
            "SPACE: Add Space",
            "C: Clear All",
            "S: Save Sentence",
            "H: Toggle Help",
            "M: Toggle Voice Mode",
            "ESC: Exit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x0, controls_y + 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.colors["text"], 1)

        settings_y = controls_y + 24 + len(lines) * 20 + 12
        if settings_y < h - 40:
            cv2.putText(frame, "SETTINGS", (x0, settings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors["primary"], 1)
            cv2.putText(
                frame,
                f"Auto-space: {'ON' if self.auto_space else 'OFF'}  (A)",
                (x0, settings_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                self.colors["text"],
                1,
            )
            cv2.putText(
                frame,
                f"Save to file: {'ON' if self.save_to_file else 'OFF'}  (F)",
                (x0, settings_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                self.colors["text"],
                1,
            )

        return frame

    def draw_speech_panel(self, frame):
        h, w = frame.shape[:2]
        uw = self.ui_width

        panel = np.zeros((h, uw, 3), dtype=np.uint8)
        panel[:] = self.colors["bg_dark"]
        frame[0:h, w - uw : w] = cv2.addWeighted(frame[0:h, w - uw : w], 0.3, panel, 0.7, 0)
        cv2.line(frame, (w - uw, 0), (w - uw, h), self.colors["primary"], 2)

        x0 = w - uw + 20
        sr = self.speech_recognizer

        cv2.putText(frame, "SPEECH MODE", (x0, 42), cv2.FONT_HERSHEY_DUPLEX, 1.05, self.colors["primary"], 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (x0, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors["text"], 1)

        st = sr.status
        if not sr.model:
            st_color = self.colors["warning"]
        elif "error" in st.lower() or "not found" in st.lower():
            st_color = self.colors["warning"]
        else:
            st_color = self.colors["success"]
        cv2.putText(frame, f"Status: {st}", (x0, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.62, st_color, 1)

        cv2.putText(frame, "RECOGNIZED TEXT", (x0, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.78, self.colors["primary"], 1)
        transcript = sr.get_text().strip()
        if transcript:
            self.wrap_text(frame, transcript, (x0, 175), uw - 40, font_scale=0.62, line_limit=14)
        else:
            cv2.putText(
                frame,
                "(Speak to see text here...)",
                (x0, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (180, 180, 180),
                1,
            )

        if time.time() < self.speech_feedback_until and self.speech_feedback_msg:
            cv2.putText(
                frame,
                self.speech_feedback_msg[:70],
                (x0, h - 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                self.colors["success"],
                1,
            )

        controls_y = h - 175
        cv2.putText(frame, "CONTROLS", (x0, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.colors["primary"], 1)
        lines = [
            "S: Save complete text (output/)",
            "C: Clear all text",
            "M: Sign mode",
            "H: Toggle help",
            "ESC: Exit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x0, controls_y + 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.colors["text"], 1)

        cv2.putText(
            frame,
            "S always writes a .txt file when text exists.",
            (x0, controls_y + 24 + len(lines) * 20 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )

        return frame

    def show_help_screen(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

        help_lines = [
            "SIGN / SPEECH — HELP",
            "",
            "SIGN MODE",
            "  Show your hand. High confidence commits letters to the current word.",
            "  ENTER finishes the word; SPACE inserts spaces; BACKSPACE deletes one character.",
            "  S saves the sentence (when Save to file is ON). C clears everything.",
            "  A toggles auto-space after each word. F toggles saving to disk.",
            "  M switches to speech; G forces sign mode; V forces speech mode.",
            "",
            "SPEECH MODE",
            "  Speak clearly. Text shows finalized phrases plus live partial.",
            "  S saves the full transcript to output/ (always works when text exists).",
            "  C clears the transcript. F toggles sign-mode save-to-file setting.",
            "",
            "Press H again to close",
        ]
        y0 = h // 10
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (w // 12, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.colors["text"], 1)

        return frame

    def save_sentence_sign(self):
        if not self.save_to_file:
            print("Save to file is OFF — press F to enable.")
            return False
        text = self.sentence.strip()
        if self.current_word:
            text = (text + (" " if text else "") + self.current_word).strip()
        if not text:
            print("Nothing to save.")
            return False

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_txt = os.path.join("output", f"sentence_{ts}.txt")
        path_json = os.path.join("output", f"sentence_{ts}.json")
        data = {
            "timestamp": ts,
            "sentence": text,
            "word_history": self.word_history,
            "settings": {"auto_space": self.auto_space},
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {path_txt}")
        return True

    def save_speech_text(self):
        """Save always works on explicit S (includes partial + finalized transcript)."""
        text = self.speech_recognizer.get_text().strip()
        if not text:
            print("Nothing to save yet — speak first, or wait for recognition.")
            self.speech_feedback_msg = "Nothing to save yet"
            self.speech_feedback_until = time.time() + 2.5
            return False

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_txt = os.path.join("output", f"speech_{ts}.txt")
        path_json = os.path.join("output", f"speech_{ts}.json")
        try:
            with open(path_txt, "w", encoding="utf-8") as f:
                f.write(text)
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump({"timestamp": ts, "text": text}, f, indent=2)
            print(f"Speech saved to {path_txt}")
            self.speech_feedback_msg = f"Saved: {os.path.basename(path_txt)}"
            self.speech_feedback_until = time.time() + 4.0
            return True
        except OSError as e:
            print(f"Save failed: {e}")
            self.speech_feedback_msg = f"Save failed: {e}"
            self.speech_feedback_until = time.time() + 3.0
            return False

    def complete_current_word(self):
        if not self.current_word:
            return
        if self.auto_space and self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        self.sentence += self.current_word
        self.word_history.append(self.current_word)
        print(f"Word added: '{self.current_word}'  Sentence: {self.sentence}")
        self.current_word = ""
        self.sign_recognizer.letter_locked = False

    def handle_sign_keys(self, key):
        if key == 13:
            self.complete_current_word()
        elif key == 32:
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
                print("Space added to sentence")
        elif key == 8:
            if self.current_word:
                self.current_word = self.current_word[:-1]
            elif self.sentence:
                self.sentence = self.sentence[:-1]
        elif key == ord("c"):
            self.current_word = ""
            self.sentence = ""
            self.word_history = []
            self.sign_recognizer.letter_locked = False
            self.speech_recognizer.clear_text()
            print("All text cleared")
        elif key == ord("s"):
            self.save_sentence_sign()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("m"):
            self.toggle_mode()
            print(f"Mode: {self.mode_names[self.mode]}")
        elif key == ord("g"):
            self.start_mode(0)
            print("Sign mode")
        elif key == ord("v"):
            self.start_mode(1)
            print("Speech mode")
        elif key == ord("a"):
            self.auto_space = not self.auto_space
            print(f"Auto-space: {'ON' if self.auto_space else 'OFF'}")
        elif key == ord("f"):
            self.save_to_file = not self.save_to_file
            print(f"Save to file: {'ON' if self.save_to_file else 'OFF'}")

    def handle_speech_keys(self, key):
        if key == ord("m"):
            self.toggle_mode()
            print(f"Mode: {self.mode_names[self.mode]}")
        elif key == ord("g"):
            self.start_mode(0)
        elif key == ord("v"):
            self.start_mode(1)
        elif key == ord("c"):
            self.speech_recognizer.clear_text()
            self.speech_feedback_msg = "Text cleared"
            self.speech_feedback_until = time.time() + 2.0
            print("Speech transcript cleared")
        elif key == ord("s"):
            self.save_speech_text()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("f"):
            self.save_to_file = not self.save_to_file
            print(f"Save to file (sign mode): {'ON' if self.save_to_file else 'OFF'}")

    def run(self):
        print("Sign Language Typing / Speech — starting (H help, ESC exit)")
        window_title = "Sign Language Typing System"

        while self.running:
            if self.mode == 0:
                ok, frame = self.sign_recognizer.cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                frame, new_char = self.sign_recognizer.process_frame(frame)
                if new_char:
                    self.current_word += new_char
                    print(f"Typed: {self.current_word}")
            else:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = self.colors["bg_dark"]

            self.calculate_fps()

            if self.mode == 0:
                frame = self.draw_sign_panel(frame)
            else:
                frame = self.draw_speech_panel(frame)

            if self.show_help:
                frame = self.show_help_screen(frame)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                if self.mode == 0 and self.save_to_file:
                    t = self.sentence.strip()
                    if self.current_word:
                        t = (t + (" " if t else "") + self.current_word).strip()
                    if t:
                        self.save_sentence_sign()
                self.running = False
            elif self.mode == 0:
                self.handle_sign_keys(key)
            else:
                self.handle_speech_keys(key)

        self.sign_recognizer.stop_camera()
        self.speech_recognizer.stop_listening()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    app = CommunicationApp()
    app.run()
