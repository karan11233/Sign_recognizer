# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
# import pyttsx3  # Import the pyttsx3 library

# # Load the trained CNN model
# model = load_model('cnn_model.h5')

# # Initialize MediaPipe hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Define labels dictionary (adjust according to your dataset)
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
#                13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
#                26: 'Hello', 27: 'Space', 28: 'Delete', 29: 'I Love You', 30: 'Sorry', 31: 'Please', 32: 'You are welcome'}

# # Initialize Text-to-Speech engine
# engine = pyttsx3.init()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# sentence = ""
# is_signing = False

# def preprocess_data(landmarks):
#     data_aux = []
#     x_ = []
#     y_ = []
    
#     for i in range(len(landmarks)):
#         x = landmarks[i].x
#         y = landmarks[i].y

#         x_.append(x)
#         y_.append(y)

#     for i in range(len(landmarks)):
#         x = landmarks[i].x
#         y = landmarks[i].y
#         data_aux.append(x - min(x_))
#         data_aux.append(y - min(y_))
    
#     return np.array(data_aux).reshape(42, 1, 1)  # Reshape to match CNN input shape

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to grab frame")
#         break

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         is_signing = True  # User is signing
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             data_aux = preprocess_data(hand_landmarks.landmark)

#             try:
#                 # Prepare the data for the CNN model
#                 data_aux = np.expand_dims(data_aux, axis=0)  # Add batch dimension
#                 prediction = model.predict(data_aux)

#                 predicted_label = int(np.argmax(prediction))
#                 predicted_character = labels_dict[predicted_label]
#                 print("Predicted character: ", predicted_character)
                
#                 # Handle special cases like Space and Delete
#                 if predicted_character == 'Space':
#                     sentence += ' '
#                 elif predicted_character == 'Delete':
#                     sentence = sentence[:-1]  # Remove last character
#                 else:
#                     sentence += predicted_character

#                 # Get bounding box coordinates
#                 x1 = int(min([lm.x for lm in hand_landmarks.landmark]) * W) - 10
#                 y1 = int(min([lm.y for lm in hand_landmarks.landmark]) * H) - 10
#                 x2 = int(max([lm.x for lm in hand_landmarks.landmark]) * W) - 10
#                 y2 = int(max([lm.y for lm in hand_landmarks.landmark]) * H) - 10

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#             except Exception as e:
#                 print("Error during prediction:", e)
        
#         # Display the updated sentence on the frame while signing
#         cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     else:
#         if is_signing:
#             # Generate and play the audio for the sentence after the user stops signing
#             engine.say(sentence)
#             engine.runAndWait()
#             is_signing = False  # Reset the signing flag after processing

#     # Ensure the sentence remains on the frame even after signing stops
#     cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('frame', frame)

#     # Check if the window was closed by the user (pressing 'q' key)
#     key = cv2.waitKey(1)
#     if key & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load the trained CNN model
model = load_model('cnn_model.h5')

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary (adjust according to your dataset)
labels_dict = {0: 'How are You', 1: 'My name is ', 2: 'Come on', 3: 'Okay', 4: 'What', 5: 'How', 6: 'When', 7: 'Thank You', 8: 'Help', 9: 'Stop', 10: 'Go', 11: 'Deaf', 12: 'Nice meeting you',
               13: 'No', 14: 'Never', 15: 'Every', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
               26: 'Hello', 27: 'Space', 28: 'Delete', 29: 'I Love You', 30: 'Sorry', 31: 'Please', 32: 'You are welcome'}

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Reduce speech speed
engine.setProperty('rate', 120)  # Adjust the rate as needed

# Initialize webcam
cap = cv2.VideoCapture(0)

sentence = ""
previous_sentence = ""
is_signing = False
last_gesture_time = time.time()  # Initialize the last gesture time

def preprocess_data(landmarks):
    data_aux = []
    x_ = []
    y_ = []
    
    for i in range(len(landmarks)):
        x = landmarks[i].x
        y = landmarks[i].y

        x_.append(x)
        y_.append(y)

    for i in range(len(landmarks)):
        x = landmarks[i].x
        y = landmarks[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))
    
    return np.array(data_aux).reshape(42, 1, 1)  # Reshape to match CNN input shape

def speak_word(word):
    engine.say(word)
    engine.runAndWait()
    time.sleep(1)  # Pause after speaking the word

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        if current_time - last_gesture_time > 2:  # Check if 2 seconds have passed
            is_signing = True  # User is signing
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = preprocess_data(hand_landmarks.landmark)

                try:
                    # Prepare the data for the CNN model
                    data_aux = np.expand_dims(data_aux, axis=0)  # Add batch dimension
                    prediction = model.predict(data_aux)

                    predicted_label = int(np.argmax(prediction))
                    predicted_character = labels_dict[predicted_label]
                    print("Predicted character: ", predicted_character)
                    
                    # Handle special cases like Space and Delete
                    if predicted_character == 'Space':
                        sentence += ' '
                    elif predicted_character == 'Delete':
                        sentence = sentence[:-1]  # Remove last character
                    else:
                        sentence += predicted_character

                    # Get bounding box coordinates
                    x1 = int(min([lm.x for lm in hand_landmarks.landmark]) * W) - 10
                    y1 = int(min([lm.y for lm in hand_landmarks.landmark]) * H) - 10
                    x2 = int(max([lm.x for lm in hand_landmarks.landmark]) * W) - 10
                    y2 = int(max([lm.y for lm in hand_landmarks.landmark]) * H) - 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    last_gesture_time = current_time  # Update the last gesture time

                except Exception as e:
                    print("Error during prediction:", e)
        
        # Display the updated sentence on the frame while signing
        cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        if is_signing:
            # Identify the new word added
            new_words = [word for word in sentence.split() if word not in previous_sentence.split()]
            if new_words:
                for word in new_words:
                    speak_word(word)
            is_signing = False  # Reset the signing flag after processing
            previous_sentence = sentence  # Update the previous sentence

    # Ensure the sentence remains on the frame even after signing stops
    cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Check if the window was closed by the user (pressing 'q' key)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


