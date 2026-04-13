import vosk
import sys
import queue
import sounddevice as sd
import json
import pyttsx3
import threading
import os


class SpeechRecognition:
    def __init__(self, model_path="vosk-model"):
        self.model_path = model_path
        self.model = None
        self.rec = None
        self.q = queue.Queue()
        self.is_running = False
        self.thread = None
        self.engine = pyttsx3.init()
        self.status = "Model not loaded"

        self._lock = threading.Lock()
        self._committed = ""  # finalized phrases from Result()
        self._partial = ""  # live partial from PartialResult()
        self._grammar_enabled = False

        if os.path.exists(self.model_path):
            try:
                # Reduce Vosk internal logging noise
                try:
                    vosk.SetLogLevel(-1)
                except Exception:
                    pass

                self.model = vosk.Model(self.model_path)
                self.rec = vosk.KaldiRecognizer(self.model, 16000)
                # Enable richer decoding details (can improve stability in some models)
                try:
                    self.rec.SetWords(True)
                except Exception:
                    pass
                try:
                    self.rec.SetPartialWords(True)
                except Exception:
                    pass

                # Optional: constrain recognition to expected phrases to improve accuracy.
                # If `phrases.json` exists (list of strings), we enable grammar mode.
                # Example file:
                # [
                #   "hello how are you",
                #   "my name is john",
                #   "i need help",
                #   "thank you"
                # ]
                phrases_path = os.path.join(os.path.dirname(__file__), "phrases.json")
                if os.path.exists(phrases_path):
                    try:
                        with open(phrases_path, "r", encoding="utf-8") as f:
                            phrases = json.load(f)
                        if isinstance(phrases, list) and all(isinstance(x, str) for x in phrases) and phrases:
                            try:
                                self.rec.SetGrammar(json.dumps(phrases))
                                self._grammar_enabled = True
                            except Exception:
                                self._grammar_enabled = False
                    except Exception:
                        self._grammar_enabled = False

                self.status = "Offline model loaded"
                if self._grammar_enabled:
                    self.status += " (phrase-boost ON)"
            except Exception as e:
                self.status = f"Error loading model: {e}"
        else:
            self.status = f"Model folder '{self.model_path}' not found"

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.is_running:
            self.q.put(bytes(indata))

    def start_listening(self):
        if not self.model:
            return False
        if self.thread is not None and self.thread.is_alive():
            return True

        self.is_running = True
        with self._lock:
            self._committed = ""
            self._partial = ""

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        return True

    def stop_listening(self):
        self.is_running = False
        try:
            self.q.put_nowait(b"\x00" * 2)
        except Exception:
            pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _listen_loop(self):
        try:
            # Lower-latency stream settings generally help responsiveness and can reduce
            # long-block artifacts (helps perceived accuracy).
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=4000,
                device=None,
                dtype="int16",
                channels=1,
                callback=self.callback,
                latency="low",
            ):
                self.status = "Listening — microphone on"
                while self.is_running:
                    try:
                        data = self.q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if not self.is_running:
                        break

                    if self.rec.AcceptWaveform(data):
                        res = json.loads(self.rec.Result())
                        text = (res.get("text") or "").strip()
                        with self._lock:
                            if text:
                                self._committed += (" " if self._committed else "") + text
                            self._partial = ""
                    else:
                        partial = json.loads(self.rec.PartialResult())
                        pt = (partial.get("partial") or "").strip()
                        with self._lock:
                            self._partial = pt
        except Exception as e:
            self.status = f"Mic error: {e}"
            self.is_running = False
        else:
            if not self.is_running:
                self.status = "Microphone stopped"

    def get_text(self):
        """Full text shown in the UI: finalized phrases plus any live partial."""
        with self._lock:
            c, p = self._committed, self._partial
        if c and p:
            return f"{c} {p}".strip()
        return (c or p).strip()

    def speak(self, text):
        if text.strip():

            def run_tts():
                self.engine.say(text)
                self.engine.runAndWait()

            threading.Thread(target=run_tts).start()

    def clear_text(self):
        with self._lock:
            self._committed = ""
            self._partial = ""
