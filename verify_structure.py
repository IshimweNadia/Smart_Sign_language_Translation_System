try:
    from sign_module import SignRecognition
    print("SignRecognition module loaded.")
except Exception as e:
    print(f"Error in sign_module: {e}")

try:
    from speech_module import SpeechRecognition
    print("SpeechRecognition module loaded.")
except Exception as e:
    print(f"Error in speech_module: {e}")

try:
    from main_app import CommunicationApp
    print("CommunicationApp class loaded.")
except Exception as e:
    print(f"Error in main_app: {e}")
