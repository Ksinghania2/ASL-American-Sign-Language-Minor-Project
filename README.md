
Real-Time ASL Alphabet Translator


This project is a real-time American Sign Language (ASL) translator built with Python, TensorFlow/Keras, and OpenCV. It uses a Convolutional Neural Network (CNN) to recognize 29 static ASL gestures from a live webcam feed and translates them into on-screen text and audible speech.

LIVE DEMO

![alt text](image.png)

This model was trained for 10 epochs on a dataset of 87,000 images, achieving a peak validation accuracy of 99.67%.


Technologies Used

	•	Python 3
	•	TensorFlow & Keras: For building and training the CNN model.
	•	OpenCV: For handling the live webcam feed and image processing.
	•	pyttsx3: For text-to-speech feedback.
	•	NumPy: For numerical operations.
	•	Google Colab: For GPU-accelerated model training.


How to Run This Project

1. Clone or Download

Download the files and place them in a single project folder.

2. Install Dependencies

You must have Python 3 installed. Navigate to the project folder in your terminal and run:

pip install -r requirements.txt

Note: This project was built using numpy==1.26.4. If you encounter a ValueError related to numpy versions, you may need to force-install the compatible version:

pip install numpy==1.26.4

3. Run the Application
With all dependencies installed, run the main script from your terminal:

python3 asl_translator.py

A webcam window will open. Press 'q' to quit the application.
