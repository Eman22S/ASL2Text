import os
import pygame
import tempfile
import shutil
import numpy as np
import cv2
import keras
import pickle
import matplotlib.pyplot as plt
import mediapipe as mp 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json
from collections import Counter
from textblob import TextBlob
from gtts import gTTS


# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
	static_image_mode=False, 
	model_complexity=1, 
	min_detection_confidence=0.75, 
	min_tracking_confidence=0.75, 
	max_num_hands=2) 


''' Load the model from the pickle file '''
# Load the dictionary from the pickle file
with open('./model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# Recreate the model architecture from the JSON string
model_json = model_dict['model_json']
model = model_from_json(model_json)

# Load the model weights
model_weights = model_dict['model_weights']
model.set_weights(model_weights)



# Function to get the bounding box of the hand
def get_bounding_box(landmarks, image_width, image_height, scale=1.0):
    x_coords = [landmark.x * image_width for landmark in landmarks.landmark]
    y_coords = [landmark.y * image_height for landmark in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Calculate the center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Calculate the size of the bounding box
    box_size = max(x_max - x_min, y_max - y_min) * scale

    # Ensure the bounding box is a square
    half_size = int(box_size // 2)

    # Calculate new min and max coordinates
    x_min_new = max(x_center - half_size, 0)
    x_max_new = min(x_center + half_size, image_width)
    y_min_new = max(y_center - half_size, 0)
    y_max_new = min(y_center + half_size, image_height)

    return x_min_new, y_min_new, x_max_new, y_max_new




# Function to predict the words

smoothing_window_size = 4
autocorrection_threshold = 3

def smooth_predictions(predictions, window_size = smoothing_window_size):
    smoothed = []
    for i in range(len(predictions) - window_size + 1):
        window = predictions[i:i + window_size]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

def remove_redundant(predictions, threshold=3):
    filtered = []
    last_char = predictions[0]
    count = 0

    for char in predictions:
        if char == last_char:
            count += 1
        else:
            if count >= threshold:
                filtered.append(last_char)
            count = 1
            last_char = char
    if count >= threshold:
            filtered.append(last_char)
        
    return filtered

def process_predicted_word(letters_list, window_size = smoothing_window_size):
    letters_list = letters_list
    #if input is too small -> don't do processing
    if len(letters_list) < window_size:
        return ''.join(letters_list).lower()
    else:
        smoothing = smooth_predictions(letters_list)
        filter_redundants= remove_redundant(smoothing)
        if len(filter_redundants) <= autocorrection_threshold:
            return ''.join(filter_redundants).lower()
        autocorrected = str(TextBlob(''.join(filter_redundants).lower()).correct())
        return autocorrected


# text to speech conversion
def text_to_speech(word, lang='en'):
    # Convert text to speech
    tts = gTTS(text=word, lang=lang)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_file = fp.name
        tts.save(temp_file)
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        # Load the audio file
        pygame.mixer.music.load(temp_file)
        # Play the audio file
        pygame.mixer.music.play()
        # Wait for the audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        # Quit the mixer and remove the temporary file after playing
        pygame.mixer.quit()
        os.remove(temp_file)

# Example usage
# word = "happy"
# text_to_speech(word)


''' Encoding the labels '''

# Example list of string values
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Convert the list to a numpy array and reshape it to a 2D array
categories_array = np.array(categories).reshape(-1, 1)
# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# Fit the encoder on the data
encoder.fit(categories_array)



''' Running the model '''

from datetime import datetime 
from IPython.display import clear_output
import threading

#For Adjusment
prediction_threshold = 0.3
wait_between_words = 3

# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 
#Words producing control
word = []
produced_words = []
last_detection_time = datetime.now()
waiting_input = True

while True:
    #to calculate FPS later
    start_time = datetime.now()
    # Read video frame by frame 
    success, img = cap.read() 
    
    # Flip the image(frame) 
    img = cv2.flip(img, 1) 
    
    # Convert BGR image to RGB image 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    # Process the RGB image 
    results = hands.process(imgRGB) 
    
    # If hands are present in image
    if results.multi_hand_landmarks:
        #Words producing control
        last_detection_time = datetime.now()
        waiting_input = False

        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            h, w, _ = img.shape
            x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255,0,0), thickness=2)
        
        #Normalize landmarks
        x_min = min([landmark.x for landmark in results.multi_hand_landmarks[0].landmark])
        y_min = min([landmark.y for landmark in results.multi_hand_landmarks[0].landmark])
        x_max = max([landmark.x for landmark in results.multi_hand_landmarks[0].landmark])
        y_max = max([landmark.y for landmark in results.multi_hand_landmarks[0].landmark])
        w = x_max - x_min
        h = y_max - y_min
        landmarks = [( (landmark.x - x_min) / w, (landmark.y - y_min) / h ) for landmark in results.multi_hand_landmarks[0].landmark]
        landmarks = list(sum(landmarks, ()))
        
        #To Show Coordinates (FOR DEVELOPING)
        '''
        formatted_list = [f"{num:.{2}f}" for num in landmarks]
        result_string1 = ", ".join(formatted_list[0:10])
        result_string2 = ", ".join(formatted_list[10:20])
        result_string3 = ", ".join(formatted_list[20:30])
        result_string4 = ", ".join(formatted_list[30:-1])
        cv2.putText(img, result_string1, (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2) 
        cv2.putText(img, result_string2, (10, 250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2) 
        cv2.putText(img, result_string3, (10, 300), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2) 
        cv2.putText(img, result_string4, (10, 350), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2) 
        '''

        #Make Prediction
        landmarks = np.array(landmarks).reshape(1, -1)
        y = model.predict(landmarks)

        #Set Confidence Threshold (only proceede to predict when meeting minimum threshold)
        if max(y[0]) > prediction_threshold:
            #Decoding Predicion
            y_decoded =  encoder.inverse_transform(y)[0][0]
            cv2.putText(img, y_decoded, (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
            cv2.putText(img, "P: {:.2f}".format(max(y[0])), (300, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
            word.append(y_decoded)
            #Show Prediction Distribution in Frame (FOR DEVELOPING)
            '''
            string_value_list = [str(v) for v in np.around(y[0], decimals=2)]
            string_value_list1 = ','.join(string_value_list[0:13])
            string_value_list2 = ','.join(string_value_list[13:-1])
            cv2.putText(img, string_value_list1, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1) 
            cv2.putText(img, string_value_list2, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1) 
            '''

    else: #NO HAND IN FRAME -> Words producing control 
        if (datetime.now() - last_detection_time).total_seconds() > wait_between_words and waiting_input == False:
            #PRODUCE WORD
            processed_word = process_predicted_word(word)
            produced_words.append(processed_word)
            threading.Thread(target=text_to_speech, args=(processed_word,)).start()
            word = []
            waiting_input = True
        elif waiting_input == False:
            cv2.putText(img, str(round((datetime.now() - last_detection_time).total_seconds(),)), (10, 350), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    #Measure FPS and Show it
    end_time = datetime.now() 
    time_difference = (end_time - start_time).total_seconds()
    cv2.putText(img, "FPS: {}".format(round(1/time_difference)), (10, 450), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
    #Print Last word
    cv2.putText(img, ' '.join(produced_words), (10, 400), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
    
	# Display Video and when 'q' is entered, destroy the window 
    cv2.imshow('Image', img) 
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):  # Replace 'q' with your preferred key
        break
    

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
