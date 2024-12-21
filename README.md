# Sign Language Recognition

This project provides a **real-time sign language recognition system** using **MediaPipe** for hand tracking and **RandomForestClassifier** for classification. The system captures sign language gestures via webcam, processes them using MediaPipe to extract hand landmarks, and classifies the gestures based on a pre-trained model.

## Features
- **Real-time Sign Language Recognition:** Uses MediaPipe's hand recognition to detect and classify hand gestures in real time.
- **Training Mode:** Allows you to input signs and store them for training.
- **Random Forest Classifier:** A machine learning model that predicts the corresponding sign language gesture based on the landmarks extracted from hand gestures.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- PyMySQL

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-recognition.git
   cd sign-language-recognition
Install the required packages:

bash
Copy code
pip install opencv-python mediapipe scikit-learn pymysql
Ensure you have a MySQL database set up with a signlanguage database containing a table signs:

sql
Copy code
CREATE TABLE signs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sign_name VARCHAR(255) NOT NULL,
    sign_encoding BLOB NOT NULL,
    signature VARCHAR(255) UNIQUE NOT NULL
);
