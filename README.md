# Smart Sign Language Translation System

##  Project Overview
The Smart Sign Language Translation System is an AI-powered application designed to enable two-way communication between hearing and deaf individuals. The system recognizes hand gestures from sign language and converts them into text and speech, while also supporting speech-to-text conversion for reverse communication.

This project bridges the communication gap using computer vision, machine learning, and speech technologies.

---

##  Features
-  Real-time hand gesture recognition
-  Conversion of sign language gestures into text
-  Text-to-speech output for audible communication
-  Speech-to-text conversion for hearing users
-  Two-way communication between deaf and hearing users
-  Live camera-based gesture detection

---

## Technologies Used
- Python
- OpenCV (Computer Vision)
- MediaPipe / Hand Tracking (if used)
- Machine Learning (Scikit-learn / TensorFlow / Keras)
- SpeechRecognition (Speech-to-Text)
- pyttsx3 / gTTS (Text-to-Speech)
- NumPy

---

## How It Works
1. The webcam captures hand gestures in real time.
2. OpenCV processes the video frames.
3. Hand landmarks are detected using a tracking model.
4. A trained machine learning model classifies the gesture.
5. The predicted gesture is displayed as text.
6. Text is converted to speech for hearing users.
7. Speech input is converted to text for deaf users.

---
