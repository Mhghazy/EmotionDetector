from sklearn.metrics import accuracy_score
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings
import pyttsx3
import time

with open('body_language_one.pkl', 'rb') as f:
    model = pickle.load(f)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')

# Initialize MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if 'female' in voice.name.lower() or 'female' in voice.id.lower():
        engine.setProperty('voice', voice.id)
        break

# Welcome message
engine.say("Hi! Welcome to Emotive. The Emotion detector software that recognizes your emotions.")
#engine.runAndWait()


cap = cv2.VideoCapture(0)  # Open the camera



# Initialize the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    last_tts_time = time.time()
    tts_interval = 5  # Interval in seconds for text-to-speech



    while cap.isOpened():
        ret, frame = cap.read()  # Read frame from the camera
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False  # Mark the image as not writeable to improve performance


        # Make detections
        results = holistic.process(image)

        # Draw face landmarks with dark purple vertices
        image.flags.writeable = True  # Mark the image as writeable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_CONTOURS,  # Use FACEMESH_CONTOURS for fewer vertices
            mp_drawing.DrawingSpec(color=(128, 0, 128, 64), thickness=1, circle_radius=1),  # Dark purple color for vertices with increased transparency
            mp_drawing.DrawingSpec(color=(80, 256, 121, 64), thickness=1, circle_radius=1)  # Green color for connections with increased transparency
        )

        # Draw pose landmarks with dark purple vertices and purple border
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=2),  # Dark purple for vertices
            mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=2)  # Purple for border
        )

        # Draw left hand landmarks with dark red vertices and connections
        mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(139, 0, 0), thickness=2, circle_radius=2),  # Dark red for left hand
            mp_drawing.DrawingSpec(color=(139, 0, 0), thickness=2, circle_radius=2)  # Dark red for connections
        )

        # Draw right hand landmarks with dark purple vertices and connections
        mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=2),  # Dark purple for right hand
            mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=2)  # Dark purple for connections
        )

        # Export coordinates
        try:
            # extract pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # extract face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())            
            # Concatenate rows
            row = pose_row + face_row
            
            # Make detections
            x = pd.DataFrame([row])
            body_language_class = model.predict(x)[0]
            body_language_prob = model.predict_proba(x)[0]
            print(body_language_class, body_language_prob)

            # Text-to-speech for detected class at intervals
            current_time = time.time()
            if current_time - last_tts_time > tts_interval:
                if body_language_class == "Happy":
                    engine.say(f"Ha ha I think you are {body_language_class}")
                elif body_language_class == "Dancing":
                    engine.say(f"Fire it up!, bom bom!, I think you are {body_language_class}")
                elif body_language_class == "Excited":
                    engine.say(f"Wow!, I think you are {body_language_class}")
                elif body_language_class == "In love":
                    engine.say(f"So romantic, I think you are {body_language_class} with someone, lucky you !")
                elif body_language_class == "Crying":
                    engine.say(f"Please don't cry, I think you are {body_language_class}")
                elif body_language_class == "Afraid":
                    engine.say(f"I smell your fear from a distance away, I think you are {body_language_class}")
                else:
                    engine.say(f"I think you are {body_language_class}")
                engine.runAndWait()
                last_tts_time = current_time

            # Grab ear coords
            coords = tuple(np.multiply(
                np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
                ), [640, 480]).astype(int))

            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS', (95,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")
            pass

        # Display the resulting frame
        cv2.imshow('MediaPipe Holistic', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()