import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# --- 1. INITIALIZE THE SPEECH ENGINE ---
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: Could not initialize text-to-speech engine: {e}")
    engine = None

# --- 2. LOAD THE TRAINED MODEL ---
try:
    # This matches the model you just trained and downloaded
    model = tf.keras.models.load_model('asl_image_model.keras')
except Exception as e:
    print(f"FATAL ERROR: Could not load model 'asl_image_model.keras'")
    print(f"Error details: {e}")
    exit()

# --- 3. DEFINE THE CLASS NAMES ---
# This list MUST match the training data's classes
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

# --- 4. START THE WEBCAM ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FATAL ERROR: Could not open webcam.")
    exit()

# --- 5. INITIALIZE STATE VARIABLES ---
last_prediction = ""
prediction_confidence_threshold = 95 # Only speak if confidence is over 95%

while True:
    # --- 6. READ A FRAME ---
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture frame.")
        break

    # --- 7. PREPROCESS THE FRAME ---
    # Resize to the model's 64x64 input size
    img = cv2.resize(frame, (64, 64))
    # Convert color from OpenCV's BGR to TensorFlow's RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Add a batch dimension (the model expects a batch, not a single image)
    img_array = tf.expand_dims(img_rgb, 0) 

    # --- 8. MAKE A PREDICTION ---
    predictions = model.predict(img_array, verbose=0) 
    score = tf.nn.softmax(predictions[0])

    # --- 9. GET AND DISPLAY THE RESULT ---
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    text = f"{predicted_class}: {confidence:.2f}%"
    
    # Draw the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 10. SPEAK THE RESULT (if engine is working) ---
    if engine and predicted_class != last_prediction and confidence > prediction_confidence_threshold:
        engine.say(predicted_class)
        engine.runAndWait()
        last_prediction = predicted_class

    # --- 11. SHOW THE FRAME ---
    cv2.imshow('Sign Language Translator', frame)

    # --- 12. EXIT CONDITION ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 13. CLEANUP ---
cap.release()
cv2.destroyAllWindows()