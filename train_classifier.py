import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib # For saving/loading models
import traceback # For detailed error printing

# --- Configuration ---
DATA_DIR = 'faces_dataset' # Directory containing subfolders for each person (e.g., ABHA/, DIKSHA/)
OUTPUT_MODEL_PATH = 'face_classifier.pkl' # Output path for the trained classifier
OUTPUT_LABEL_ENCODER_PATH = 'label_encoder.pkl' # Output path for the label encoder

# --- Load FaceNet model ---
try:
    print("Loading FaceNet model...")
    # Using 'vggface2' pretrained weights, set to evaluation mode
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("FaceNet model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load FaceNet model. Make sure you have 'facenet_pytorch' installed correctly and internet access for initial download (if not cached).")
    print(f"Details: {e}")
    traceback.print_exc()
    exit() # Exit if the core model cannot be loaded

# --- Load Haar Cascade for face detection ---
try:
    # CORRECTED TYPO: cv2.data.haascades -> cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("ERROR: Haar Cascade XML file not loaded. Ensure 'haarcascade_frontalface_default.xml' is in OpenCV data path.")
        exit()
    print("Haar Cascade classifier loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load Haar Cascade classifier: {e}")
    traceback.print_exc()
    exit()

# --- Function to get FaceNet embedding from an image ---
def get_face_embedding(image_path):
    """
    Detects a face in the image, crops it, and returns its FaceNet embedding.
    image_path: Path to the image file.
    Returns the embedding (np.array) or None if no face is detected or an error occurs.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None

        # Convert to RGB (FaceNet expects RGB, OpenCV reads BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"Warning: No face detected in {image_path}. Skipping.")
            return None

        # Take the first detected face (you might want to handle multiple faces)
        x, y, w, h = faces[0]
        face_img = img_rgb[y:y+h, x:x+w]

        if face_img.size == 0:
            print(f"Warning: Cropped face from {image_path} is empty. Skipping.")
            return None

        # Resize the face image to 160x160 as required by FaceNet model
        face_resized = cv2.resize(face_img, (160, 160))

        # Convert to PyTorch tensor format (C, H, W) and normalize to [-1, 1]
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float()
        face_normalized = (face_tensor - 127.5) / 128.0
        face_normalized = face_normalized.unsqueeze(0) # Add batch dimension

        # Get embedding
        with torch.no_grad(): # Disable gradient calculation for inference
            embedding = model(face_normalized).numpy().flatten()
        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return None

# --- Collect embeddings and labels ---
embeddings = []
labels = []
label_names = [] # To store actual names/roll_numbers for LabelEncoder

print(f"\nScanning for face images in: {DATA_DIR}")
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add subfolders with images.")
    exit()

for person_name in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person_name)
    if os.path.isdir(person_dir):
        print(f"Processing images for: {person_name}")
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_name)
                embedding = get_face_embedding(img_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(person_name) # Use person_name (e.g., roll number) as the label
                    print(f"  - Added embedding for {img_name}")
                else:
                    print(f"  - Failed to get embedding for {img_name}")

if not embeddings:
    print("No embeddings found. Training aborted. Please check your data directory and image files.")
    exit()

embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"\nTotal embeddings collected: {len(embeddings)}")
print(f"Unique labels found: {np.unique(labels)}")

# --- Encode labels ---
print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
print(f"Label classes: {le.classes_}")

# --- Train SVC classifier ---
print("Training SVC classifier...")
# Using a linear SVM, you can experiment with other kernels or classifiers
classifier_model = SVC(kernel='linear', probability=True)
classifier_model.fit(embeddings, y_encoded)
print("Classifier training complete.")

# --- Save the trained classifier and label encoder ---
try:
    joblib.dump(classifier_model, OUTPUT_MODEL_PATH)
    joblib.dump(le, OUTPUT_LABEL_ENCODER_PATH)
    print(f"\nTrained classifier saved to: {OUTPUT_MODEL_PATH}")
    print(f"Label encoder saved to: {OUTPUT_LABEL_ENCODER_PATH}")
    print("\nTraining process completed successfully. You can now run your Flask application (`server.py`).")
except Exception as e:
    print(f"ERROR: Failed to save trained models: {e}")
    traceback.print_exc()
