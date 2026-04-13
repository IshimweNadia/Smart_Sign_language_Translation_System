try:
    import vosk
    print("vosk installed")
except ImportError:
    print("vosk NOT installed")

try:
    import pyttsx3
    print("pyttsx3 installed")
except ImportError:
    print("pyttsx3 NOT installed")

try:
    import pyaudio
    print("pyaudio installed")
except ImportError:
    print("pyaudio NOT installed")
