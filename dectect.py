import tensorflow as tf
import cv2
import numpy as np

class ScrewDetector:
    def __init__(self, model_path='screw_detection_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.img_height = 224
        self.img_width = 224

    def preprocess_image(self, image):
        # Resize image to match model's expected sizing
        resized = cv2.resize(image, (self.img_height, self.img_width))
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)

    def predict(self, image):
        preprocessed = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed)
        return prediction[0][0], "Screw Present" if prediction[0] > 0.5 else "Screw Missing"

    def find_screw_region(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        return None

def process_video():
    detector = ScrewDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make prediction
        confidence, result = detector.predict(frame)
        
        # Find ROI
        roi = detector.find_screw_region(frame)
        if roi:
            x, y, w, h = roi
            # Draw rectangle
            color = (0, 255, 0) if "Present" in result else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Add text to frame
        cv2.putText(frame, 
                    f"Status: {result} ({confidence:.2f})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0) if "Present" in result else (0, 0, 255), 
                    2)

        cv2.imshow('Screw Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    detector = ScrewDetector()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    confidence, result = detector.predict(image)
    
    # Find ROI
    roi = detector.find_screw_region(image)
    if roi:
        x, y, w, h = roi
        # Draw rectangle
        color = (0, 255, 0) if "Present" in result else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    # Add text to image
    cv2.putText(image, 
                f"Status: {result} ({confidence:.2f})", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0) if "Present" in result else (0, 0, 255), 
                2)

    cv2.imshow('Screw Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Screw Detection Tool')
    parser.add_argument('--image', type=str, help='Path to image file (optional)')
    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    else:
        process_video()
