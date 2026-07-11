import cv2
import numpy as np
import os
import sys

# Suppress noisy TensorFlow initialization alerts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def audio_worker(text_queue):
    """Runs speech tasks in an isolated OS process boundary to eliminate macOS threading conflicts."""
    import pyttsx3
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        while True:
            text = text_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
    except Exception:
        pass

class ASLTranslator:
    def __init__(self, queue):
        print("[INIT] Booting ASL High-Precision Prediction Engine...")
        self.queue = queue
        
        try:
            self.model = tf.keras.models.load_model('asl_image_model.keras')
            print("[SUCCESS] Deep learning neural network weights compiled cleanly.")
        except Exception as e:
            print(f"[FATAL] System failure loading model asset: {e}")
            sys.exit(1)

        # Strict Alphabetical ASCII Order mapped to dataset folder layout
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                            'del', 'nothing', 'space']
        
        self.last_prediction = ""
        
        # ⚡ CONFIGURATION LOCKS FOR HIGH ACCURACY
        self.confidence_threshold = 92.0  # Require high certainty before triggering audio
        self.prediction_history = []
        self.history_window_size = 15     # Accumulate 15 frames to completely smooth out switching letters

    def start_pipeline(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FATAL] Video capture interface unreachable.")
            sys.exit(1)

        print("\n" + "="*65)
        print("  SYSTEM LIVE: Target your hand inside the green square box.")
        print("  Press 'q' inside the display window to exit safely.")
        print("="*65 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Enforce a perfect square crop box matching training image aspect ratios
            box_size = 260
            x1 = int(w * 0.55)
            y1 = int(h * 0.15)
            x2 = x1 + box_size
            y2 = y1 + box_size

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                # 1. Resize directly to the dataset target scale (64x64)
                img_resized = cv2.resize(roi, (64, 64))
                
                # 2. Correct color channels from OpenCV BGR to TensorFlow RGB layout
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # 3. ⚡ FIXED RESCALING CONVERSION: Cast to standard float32 array
                img_float = img_rgb.astype(np.float32)
                
                # If your model requires raw integers, use img_rgb instead. 
                # If it expects normalized vectors, divide by 255.0 here.
                # Standard Keras vision datasets require explicit normalization batch mapping:
                img_normalized = img_float / 255.0 if not any(isinstance(l, tf.keras.layers.Rescaling) for l in self.model.layers) else img_float
                
                img_tensor = np.expand_dims(img_normalized, axis=0)

                # 4. Run Inference Pipeline
                predictions = self.model.predict(img_tensor, verbose=0)[0]
                
                # Apply stable mathematical Softmax transformation on raw array logit spaces
                exp_preds = np.exp(predictions - np.max(predictions))
                probabilities = exp_preds / exp_preds.sum()
                
                raw_idx = np.argmax(probabilities)
                raw_class = self.class_names[raw_idx]
                raw_confidence = probabilities[raw_idx] * 100.0

                # 5. ⚡ HISTORICAL SMOOTHING FILTER: Eliminate jumping alphabet flicker
                self.prediction_history.append(raw_class)
                if len(self.prediction_history) > self.history_window_size:
                    self.prediction_history.pop(0)

                # Extract the statistical mode (most consistently seen letter over the last 15 frames)
                stabilized_class = max(set(self.prediction_history), key=self.prediction_history.count)
                
                # Bind confidence outputs to verified stabilized prediction state
                display_confidence = raw_confidence if stabilized_class == raw_class else 0.0

                # Render precision outputs directly to window UI layer
                ui_text = f"ASL Sign: {stabilized_class} ({display_confidence:.1f}%)"
                cv2.putText(frame, ui_text, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Dispatch async text-to-speech task when a stable new sign is held
                if (stabilized_class != self.last_prediction and 
                    display_confidence >= self.confidence_threshold and 
                    stabilized_class not in ['nothing', 'del', 'space']):
                    
                    self.queue.put(stabilized_class)
                    self.last_prediction = stabilized_class

            cv2.imshow('ASL Minor Project Production Engine', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.queue.put(None)
        print("[SHUTDOWN] Frame metrics engine destroyed safely.")

if __name__ == "__main__":
    from multiprocessing import Process, Queue
    
    communication_queue = Queue()
    audio_process = Process(target=audio_worker, args=(communication_queue,))
    audio_process.start()
    
    translator = ASLTranslator(communication_queue)
    translator.start_pipeline()
    
    audio_process.join()