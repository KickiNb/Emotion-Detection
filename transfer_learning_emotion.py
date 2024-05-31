import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model_path = r"C:\Users\kicki\OneDrive\Documents\ec_utbildning\deeplearning\Kod\Face_cnn\final_model_vgg16_with_more_layers.h5"
model = load_model(model_path)

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier(r'C:\Users\kicki\OneDrive\Documents\ec_utbildning\deeplearning\Kod\Face_cnn\haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Print predictions of what the model is predicting
            print(f"Prediction: {prediction}, Label: {label}")
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
