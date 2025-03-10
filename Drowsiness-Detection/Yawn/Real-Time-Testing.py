import os
import cv2
import numpy as np
import dlib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Force TensorFlow to use CPU only (Prevents GPU access delay)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU

# Paths
base_folder = r"E:\drowsiness"
folders = ["input", "grayscale", "crop", "landmarked", "final"]

# Create necessary directories
for folder in folders:
    os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

model_path = r"E:\yawn_models\yawn_model.h5"
predictor_path = r"C:\Users\dharu\PycharmProjects\pythonProject1\.venv\shape_predictor\shape_predictor_68_face_landmarks.dat"

# Load model and face detector
model = load_model(model_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Thresholds for detection
RAW_THRESHOLD = 0.6  # CNN model threshold
LIP_DISTANCE_THRESHOLD = 25.0  # Lip distance threshold

def capture_image():
    logitech_cam_index = 1
    cap = cv2.VideoCapture(logitech_cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer delay

    if not cap.isOpened():
        print("Error: Could not access external webcam! Trying default camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No camera detected!")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image!")
            continue

        cv2.imshow("Press 'C' to capture, 'Q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            image_path = os.path.join(base_folder, "input", "captured_img.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved at: {image_path}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_path

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_path = os.path.join(base_folder, "grayscale", "grayscale_img.jpg")
    cv2.imwrite(grayscale_path, gray)

    faces = detector(gray, 1)
    if len(faces) == 0:
        raise ValueError("No face detected!")

    face = faces[0]
    cropped_face = gray[face.top():face.bottom(), face.left():face.right()]
    cropped_face = cv2.resize(cropped_face, (64, 64))
    cropped_face_path = os.path.join(base_folder, "crop", "cropped_img.jpg")
    cv2.imwrite(cropped_face_path, cropped_face)

    landmarks = predictor(gray, face)
    mouth_landmarks = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])

    for (x, y) in mouth_landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    landmarked_path = os.path.join(base_folder, "landmarked", "landmarked_img.jpg")
    cv2.imwrite(landmarked_path, image)

    # Convert grayscale image to BGR before overlaying
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    final_image = cv2.addWeighted(image, 0.7, gray_bgr, 0.3, 0)
    final_image_path = os.path.join(base_folder, "final", "final_img.jpg")
    cv2.imwrite(final_image_path, final_image)

    upper_lip = np.mean(mouth_landmarks[0:6], axis=0)
    lower_lip = np.mean(mouth_landmarks[6:12], axis=0)
    lip_distance = np.linalg.norm(upper_lip - lower_lip)
    print(f"Lip Distance: {lip_distance:.2f}")

    cropped_face = cropped_face / 255.0
    return cropped_face.reshape(1, 64, 64, 1), lip_distance

captured_image_path = capture_image()
if captured_image_path:
    input_data, lip_distance = preprocess_image(captured_image_path)
    raw_prediction = model.predict(input_data, verbose=0)[0][0]
    print(f"Raw Model Prediction: {raw_prediction:.2f}")

    if raw_prediction > RAW_THRESHOLD and lip_distance > LIP_DISTANCE_THRESHOLD:
        print("Yawning Detected 😮")
    else:
        print("Not Yawning 😊")
