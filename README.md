# ASL to Text Generator - Machine Learning Project
This project implements a machine learning model for converting American Sign Language (ASL) signs to text, with the ability to speak the predicted text aloud.

## Project Overview
This project utilizes computer vision and deep learning techniques to achieve ASL to text conversion. A hand detection and pose estimation model (MediaPipe) extracts features from video frames of ASL signs. These features are then fed into a deep learning model (built with TensorFlow and Keras) to predict the corresponding text. Additionally, the project integrates text-to-speech functionality using the gtts and pygame libraries, allowing the predicted text to be spoken aloud.

## Dependencies
This project requires the following libraries:

scikit-learn (https://scikit-learn.org/)

TensorFlow (https://www.tensorflow.org/)

Keras (https://keras.io/)

MediaPipe (https://ai.google.dev/edge/mediapipe/solutions/guide)

OpenCV (potentially, for video processing) (https://opencv.org/)

gTTS (https://gtts.readthedocs.io/en/latest/)

Pygame (https://www.pygame.org/)

## ASL Examples
![ASL Example](ASL-Example.png)

## DEMO
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/FGRAcczxRJw/0.jpg)](https://www.youtube.com/watch?v=FGRAcczxRJw)


## ASL Dataset Used
https://drive.google.com/file/d/1tpjjLkuqXWF9RuKMq04YgQhp9UjMcx4D/view
