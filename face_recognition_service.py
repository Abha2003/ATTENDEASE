import cv2
import numpy as np
import os
import torch
import joblib
import time
import traceback
from PIL import Image
import io
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from facenet_pytorch import InceptionResnetV1

# --- Global Model and Classifier Variables ---
model = None
face_cascade = None
classifier = None
label_encoder = None

# --- Configuration Constants ---
MIN_FACE_AREA_PIXELS = 30 * 30
CONFIDENCE_THRESHOLD = 0.70 # Adjusted slightly for potentially more robust recognition (was 0.75)
STRICT_DISTANCE_THRESHOLD = 0.65 # Adjusted slightly for potentially more robust recognition (was 0.6)
DUPLICATE_FACE_THRESHOLD = 0.7 # For checking if a new registration is too close to an existing one

# --- Paths ---
FACE_DATA_DIR = 'face_data'
CLASSIFIER_PATH = 'face_classifier.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
HAAR_CASCADE_NAME = 'haarcascade_frontalface_default.xml' # Defined as constant

# Ensure directories exist
os.makedirs(FACE_DATA_DIR, exist_ok=True)
print(f"DEBUG: FACE_DATA_DIR '{FACE_DATA_DIR}' ensured.")

def _load_facenet_model():
    """Loads the FaceNet InceptionResnetV1 model."""
    global model
    if model is None:
        print("  LOADING FACENET INCEPTIONRESNETV1 (PRETRAINED='VGGFACE2')...")
        try:
            model = InceptionResnetV1(pretrained='vggface2').eval()
            if isinstance(model, InceptionResnetV1):
                print("  FACENET INCEPTIONRESNETV1 MODEL LOADED SUCCESSFULLY.")
            else:
                print("  WARNING: FACENET INCEPTIONRESNETV1 MODEL DID NOT RETURN AN EXPECTED OBJECT.")
                model = None
        except ImportError:
            print("CRITICAL ERROR: FACENET_PYTORCH NOT INSTALLED. Please install it using 'pip install facenet_pytorch'.")
            print("  Command: pip install facenet_pytorch")
            model = None
        except Exception as e:
            print(f"CRITICAL ERROR: FAILED TO LOAD FACENET INCEPTIONRESNETV1 MODEL: {e}")
            traceback.print_exc()
            model = None
    return model

def _load_haar_cascade():
    """Initializes and loads the Haar Cascade classifier."""
    global face_cascade
    if face_cascade is None:
        haar_cascade_path = os.path.join(cv2.data.haarcascades, HAAR_CASCADE_NAME)
        print(f"  ATTEMPTING TO LOAD HAAR CASCADE FROM PATH: {haar_cascade_path}")
        try:
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if face_cascade.empty():
                print(f"  ERROR: HAAR CASCADE XML FILE '{HAAR_CASCADE_NAME}' NOT LOADED.")
                print(f"  Please ensure it exists at: {haar_cascade_path}")
            else:
                print("  HAAR CASCADE CLASSIFIER LOADED SUCCESSFULLY.")
        except Exception as e:
            print(f"CRITICAL ERROR: FAILED TO LOAD HAAR CASCADE CLASSIFIER: {e}")
            traceback.print_exc()
            face_cascade = None
    return face_cascade

def _load_classifier_and_label_encoder():
    """Loads the trained classifier and label encoder."""
    global classifier, label_encoder
    if classifier is None or label_encoder is None:
        print("  LOADING TRAINED CLASSIFIER (FACE_CLASSIFIER.PKL) AND LABEL ENCODER (LABEL_ENCODER.PKL)...")
        try:
            classifier = joblib.load(CLASSIFIER_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            print("  TRAINED CLASSIFIER AND LABEL ENCODER LOADED SUCCESSFULLY.")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: '{CLASSIFIER_PATH}' or '{LABEL_ENCODER_PATH}' NOT FOUND.")
            print("  Please run 'train_face_classifier()' script first to generate these files.")
            classifier = None
            label_encoder = None
        except Exception as e:
            print(f"CRITICAL ERROR: FAILED TO LOAD TRAINED CLASSIFIER OR LABEL ENCODER: {e}")
            traceback.print_exc()
            classifier = None
            label_encoder = None
    return classifier, label_encoder

def init_face_recognition_models():
    """Initializes all required models for face recognition."""
    print("\nINITIALIZING FACE RECOGNITION MODELS...")
    model_loaded = _load_facenet_model() is not None
    cascade_loaded = _load_haar_cascade() is not None
    classifier_loaded, encoder_loaded = _load_classifier_and_label_encoder()
    
    all_loaded = model_loaded and cascade_loaded and classifier_loaded is not None and encoder_loaded is not None
    print("--- FACE RECOGNITION MODEL INITIALIZATION COMPLETE ---")
    print(f"DEBUG SERVICE: FACENET MODEL LOADED: {model_loaded}")
    print(f"DEBUG SERVICE: HAAR CASCADE LOADED: {cascade_loaded}")
    print(f"DEBUG SERVICE: CLASSIFIER LOADED: {classifier_loaded is not None}")
    print(f"DEBUG SERVICE: LABEL ENCODER LOADED: {encoder_loaded is not None}")
    
    if not all_loaded:
        print("WARNING: Not all face recognition components are ready. Functionality may be limited.")
    return all_loaded

def get_largest_face_coordinates(image_array_rgb):
    """
    Detects faces in the image_array_rgb using Haar Cascade and returns
    coordinates (x, y, w, h) of the largest face, or None if no face is found.
    """
    if face_cascade is None:
        print("ERROR: HAAR CASCADE IS NOT INITIALIZED FOR FACE DETECTION IN SERVICE. Cannot detect faces.")
        return None

    try:
        gray_image = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2GRAY)
        # Using a slightly lower minNeighbors for potentially more detections, adjust if too many false positives
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)) # Increased minSize for better quality
        print(f"DEBUG FACE_DETECT: Found {len(faces)} faces.")

        if len(faces) == 0:
            return None

        largest_face = None
        max_area = 0
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)
            
        if largest_face:
            print(f"DEBUG FACE_DETECT: Largest face found at (x,y,w,h): {largest_face}, Area: {max_area}")
            if max_area < MIN_FACE_AREA_PIXELS:
                print(f"DEBUG FACE_DETECT: Largest face area {max_area} is below minimum threshold {MIN_FACE_AREA_PIXELS}. Rejecting.")
                return None
        return largest_face
    except Exception as e:
        print(f"ERROR DURING FACE DETECTION/COORDINATE EXTRACTION IN SERVICE: {e}")
        traceback.print_exc()
        return None

def get_facenet_embedding_from_cv2_rgb(rgb_img_array):
    """
    Processes an RGB image array (H, W, C) to get a FaceNet embedding.
    Assumes the global 'model' is loaded.
    """
    if model is None:
        raise RuntimeError("FACENET MODEL IS NOT LOADED FOR EMBEDDING GENERATION IN SERVICE. Cannot generate embedding.")

    try:
        face_resized = cv2.resize(rgb_img_array, (160, 160))
        face_transposed = np.transpose(face_resized, (2, 0, 1)) # HWC to CHW
        face_tensor = torch.tensor(face_transposed, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        face_normalized = (face_tensor - 127.5) / 128.0 # Normalize to [-1, 1]

        with torch.no_grad():
            embedding = model(face_normalized).numpy().flatten()
        print("DEBUG EMBEDDING: FaceNet embedding generated successfully.")
        return embedding
    except Exception as e:
        print(f"ERROR DURING FACENET EMBEDDING GENERATION IN SERVICE: {e}")
        traceback.print_exc()
        return None

def detect_brightness_issue(rgb_image_array):
    """
    Analyzes an RGB image array for brightness/contrast issues.
    Returns a string indicating the issue type if found, otherwise None.
    """
    if rgb_image_array.size == 0:
        return "EMPTY_IMAGE"

    gray_image = cv2.cvtColor(rgb_image_array, cv2.COLOR_RGB2GRAY)
    
    mean_val, std_dev_val = cv2.meanStdDev(gray_image)
    mean_intensity = mean_val[0][0]
    std_dev_intensity = std_dev_val[0][0]

    BRIGHTNESS_THRESHOLD_HIGH = 200 # Values above this are too bright
    BRIGHTNESS_THRESHOLD_LOW = 50   # Values below this are too dark
    CONTRAST_THRESHOLD = 20

    if mean_intensity > BRIGHTNESS_THRESHOLD_HIGH:
        return "TOO_BRIGHT"
    elif mean_intensity < BRIGHTNESS_THRESHOLD_LOW:
        return "TOO_DARK"
    elif std_dev_intensity < CONTRAST_THRESHOLD:
        return "LOW_CONTRAST"
    return None

def verify_student_registration_image(pil_image_rgb, roll_no_to_register, class_to_register):
    """
    Processes a PIL RGB image for registration:
    1. Detects the largest face.
    2. Generates FaceNet embedding.
    3. Checks for duplicate faces against existing students.
    Returns (True, embedding_array, message) on success, (False, None, error_message) on failure.
    """
    print(f"\n--- VERIFYING REGISTRATION IMAGE for ROLL NO: {roll_no_to_register}, Class: {class_to_register} ---")
    if not init_face_recognition_models():
        return False, None, "FACE RECOGNITION SYSTEM NOT FULLY LOADED. CANNOT REGISTER STUDENT. Please check server logs."

    try:
        cv2_rgb_image = np.array(pil_image_rgb).astype(np.uint8) 
        print(f"DEBUG REG_IMG: Converted PIL image to CV2 RGB array. Shape: {cv2_rgb_image.shape}")
        
        face_coords = get_largest_face_coordinates(cv2_rgb_image)

        if face_coords is None:
            return False, None, "NO FACE DETECTED IN THE PHOTO. PLEASE TRY AGAIN WITH A CLEARER PHOTO."

        x, y, w, h = face_coords
        
        # Check face dimensions after detection
        if w <= 0 or h <= 0 or (w * h) < MIN_FACE_AREA_PIXELS:
            print(f"DEBUG REG_IMG: INVALID FACE ROI DIMENSIONS: W={w}, H={h}, Area={w*h}. Minimum required: {MIN_FACE_AREA_PIXELS}")
            return False, None, "FACE NOT CLEAR: TOO SMALL OR PARTIALLY VISIBLE. PLEASE ENSURE YOUR WHOLE FACE IS VISIBLE."

        face_roi_rgb = cv2_rgb_image[y:y+h, x:x+w]
        print(f"DEBUG REG_IMG: Cropped face ROI. Shape: {face_roi_rgb.shape}")
        
        if face_roi_rgb.size == 0:
            return False, None, "CROPPED FACE REGION IS EMPTY. PLEASE TRY A DIFFERENT PHOTO."
        
        brightness_issue_type = detect_brightness_issue(face_roi_rgb)
        if brightness_issue_type:
            msg = ""
            if brightness_issue_type == "TOO_DARK": msg = "IMAGE TOO DARK. PLEASE USE BETTER LIGHTING."
            elif brightness_issue_type == "TOO_BRIGHT": msg = "IMAGE TOO BRIGHT. PLEASE REDUCE LIGHTING."
            elif brightness_issue_type == "LOW_CONTRAST": msg = "IMAGE HAS LOW CONTRAST. PLEASE ADJUST LIGHTING OR BACKGROUND."
            else: msg = "IMAGE QUALITY ISSUE DETECTED. PLEASE TRY AGAIN."
            print(f"DEBUG REG_IMG: Image quality issue: {brightness_issue_type}. Message: {msg}")
            return False, None, msg

        face_embedding = get_facenet_embedding_from_cv2_rgb(face_roi_rgb)
        
        if face_embedding is None:
            return False, None, "FAILED TO GENERATE FACE EMBEDDING. PLEASE TRY AGAIN WITH A CLEARER PHOTO."
        print(f"DEBUG REG_IMG: Generated embedding of shape: {face_embedding.shape}")

        # Check for duplicate faces against existing students (using stored .npy files)
        existing_face_files = [f for f in os.listdir(FACE_DATA_DIR) if f.endswith('.npy')]
        print(f"DEBUG REG_IMG: Checking against {len(existing_face_files)} existing face embeddings for duplicates.")
        for existing_file in existing_face_files:
            existing_roll_no = os.path.splitext(existing_file)[0].upper()
            existing_npy_path = os.path.join(FACE_DATA_DIR, existing_file)
            
            try:
                stored_embedding = np.load(existing_npy_path)
                distance = np.linalg.norm(face_embedding - stored_embedding)
                
                if distance < DUPLICATE_FACE_THRESHOLD:
                    print(f"DEBUG REG_IMG: DUPLICATE FACE DETECTED (DISTANCE: {distance:.4f}) FOR EXISTING ROLL NO: {existing_roll_no} (Threshold: {DUPLICATE_FACE_THRESHOLD}).")
                    return False, None, f"THIS FACE IS TOO SIMILAR TO AN ALREADY REGISTERED FACE (ROLL NO: {existing_roll_no})."
            except Exception as e:
                print(f"DEBUG REG_IMG: ERROR CHECKING DUPLICATE FOR {existing_roll_no} from {existing_npy_path}: {e}")
                traceback.print_exc()
                continue # Continue checking other files even if one fails

        print("DEBUG REG_IMG: No significant duplicate face found.")
        return True, face_embedding, "FACE PROCESSED SUCCESSFULLY."

    except Exception as e:
        print(f"CRITICAL ERROR DURING FACE PROCESSING FOR REGISTRATION IN SERVICE: {e}")
        traceback.print_exc()
        return False, None, f"AN UNEXPECTED ERROR OCCURRED DURING FACE PROCESSING: {e}"


def recognize_face_from_base64_image(image_data_base64):
    """
    Processes a base64 encoded image for face recognition.
    Returns (True, {'roll_no': predicted_roll_no, 'embedding': current_embedding}, message)
    on successful recognition, (False, None, error_message) on failure or no match.
    """
    print("\n--- RECOGNIZING FACE FROM BASE64 IMAGE ---")
    if not init_face_recognition_models():
        return False, None, "FACE RECOGNITION SYSTEM NOT FULLY LOADED. PLEASE CHECK SERVER LOGS."

    try:
        if "," in image_data_base64:
            image_data_base64 = image_data_base64.split(",")[1]
        
        image_bytes = io.BytesIO(base64.b64decode(image_data_base64))
        pil_image = Image.open(image_bytes).convert('RGB')
        
        cv2_rgb_image = np.array(pil_image).astype(np.uint8) 
        print(f"DEBUG RECOGNIZE_BASE64: Converted base64 to CV2 RGB. Shape: {cv2_rgb_image.shape}")
        
        face_coords = get_largest_face_coordinates(cv2_rgb_image)
        
        if face_coords is None:
            return False, None, "NO FACE DETECTED IN THE CAPTURED IMAGE. PLEASE TRY AGAIN."

        x, y, w, h = face_coords
        
        if w <= 0 or h <= 0 or (w * h) < MIN_FACE_AREA_PIXELS:
            print(f"DEBUG RECOGNIZE_BASE64: INVALID FACE ROI DIMENSIONS: W={w}, H={h}, Area={w*h}. Threshold={MIN_FACE_AREA_PIXELS}")
            return False, None, "FACE NOT CLEAR: TOO SMALL OR PARTIALLY VISIBLE."
        
        face_roi_rgb = cv2_rgb_image[y:y+h, x:x+w]
        print(f"DEBUG RECOGNIZE_BASE64: Cropped face ROI. Shape: {face_roi_rgb.shape}")
        
        if face_roi_rgb.size == 0:
            return False, None, "CROPPED FACE IMAGE IS EMPTY."
        
        brightness_issue_type = detect_brightness_issue(face_roi_rgb)
        if brightness_issue_type:
            msg = ""
            if brightness_issue_type == "TOO_DARK": msg = "FACE NOT CLEAR: IMAGE TOO DARK. USE BETTER LIGHTING."
            elif brightness_issue_type == "TOO_BRIGHT": msg = "FACE NOT CLEAR: IMAGE TOO BRIGHT. REDUCE LIGHTING."
            elif brightness_issue_type == "LOW_CONTRAST": msg = "FACE NOT CLEAR: IMAGE HAS LOW CONTRAST. ADJUST LIGHTING."
            else: msg = "FACE NOT CLEAR DUE TO IMAGE QUALITY ISSUES."
            print(f"DEBUG RECOGNIZE_BASE64: Image quality issue: {brightness_issue_type}. Message: {msg}")
            return False, None, msg

        current_embedding = get_facenet_embedding_from_cv2_rgb(face_roi_rgb)
        if current_embedding is None:
            return False, None, "FAILED TO GENERATE FACE EMBEDDING. PLEASE TRY AGAIN."
        print(f"DEBUG RECOGNIZE_BASE64: Generated embedding of shape: {current_embedding.shape}")
        
        # Predict using the trained classifier
        if classifier is None or label_encoder is None:
            print("DEBUG RECOGNIZE_BASE64: Classifier/Label Encoder not loaded, attempting to load again.")
            _load_classifier_and_label_encoder()
            if classifier is None or label_encoder is None:
                return False, None, "CLASSIFIER NOT LOADED. RUN TRAINING FIRST."

        prediction = classifier.predict([current_embedding])
        probabilities = classifier.predict_proba([current_embedding])[0]
        max_probability = np.max(probabilities)
        predicted_roll_no = label_encoder.inverse_transform(prediction)[0].upper()
        
        print(f"DEBUG RECOGNIZE_BASE64: CLASSIFIER PREDICTED ROLL NO: {predicted_roll_no} (CONFIDENCE: {max_probability:.2f})")

        if max_probability < CONFIDENCE_THRESHOLD:
            print(f"DEBUG RECOGNIZE_BASE64: CLASSIFIER CONFIDENCE {max_probability:.2f} IS BELOW THRESHOLD {CONFIDENCE_THRESHOLD}. REJECTED.")
            return False, None, "FACE NOT RECOGNIZED WITH SUFFICIENT CONFIDENCE. PLEASE TRY AGAIN."

        # Verify with FaceNet distance against the stored embedding for the predicted roll_no
        npy_path = os.path.join(FACE_DATA_DIR, f'{predicted_roll_no}.npy')
        print(f"DEBUG RECOGNIZE_BASE64: Checking stored embedding at: {npy_path}")

        if os.path.exists(npy_path):
            try:
                stored_embedding = np.load(npy_path)
                distance = np.linalg.norm(current_embedding - stored_embedding)
                
                print(f"DEBUG RECOGNIZE_BASE64: COMPARING WITH STORED EMBEDDING FOR {predicted_roll_no}. DISTANCE: {distance:.4f} (THRESHOLD: {STRICT_DISTANCE_THRESHOLD})")

                if distance < STRICT_DISTANCE_THRESHOLD:
                    return True, {'roll_no': predicted_roll_no, 'embedding': current_embedding}, f"RECOGNIZED {predicted_roll_no} (CONFIDENCE: {max_probability:.2f}, DISTANCE: {distance:.2f})"
                else:
                    print(f"DEBUG RECOGNIZE_BASE64: PREDICTED {predicted_roll_no} BUT DISTANCE {distance:.4f} > {STRICT_DISTANCE_THRESHOLD}. NOT A STRONG ENOUGH MATCH.")
                    return False, None, "FACE NOT RECOGNIZED. PLEASE TRY AGAIN."
            except Exception as e:
                print(f"DEBUG RECOGNIZE_BASE64: ERROR LOADING OR COMPARING EMBEDDING FOR {predicted_roll_no} from {npy_path}: {e}")
                traceback.print_exc()
                return False, None, "ERROR DURING FACE VERIFICATION. PLEASE RETRY."
        else:
            print(f"DEBUG RECOGNIZE_BASE64: .NPY FILE NOT FOUND FOR PREDICTED ROLL NO {predicted_roll_no}. CANNOT PERFORM EMBEDDING DISTANCE VERIFICATION.")
            return False, None, "FACE NOT RECOGNIZED. ENROLLMENT DATA MISSING OR CORRUPT."
    except Exception as e:
        print(f"CRITICAL ERROR: GENERAL ERROR IN RECOGNIZE_FACE_FROM_BASE64_IMAGE: {e}")
        traceback.print_exc()
        return False, None, f"AN UNEXPECTED ERROR OCCURRED: {e}"

def recognize_face_from_image_path(image_path):
    """
    Processes a saved image from a file path for face recognition.
    Returns (True, {'roll_no': predicted_roll_no, 'embedding': current_embedding}, message)
    on successful recognition, (False, None, error_message) on failure or no match.
    """
    print(f"\n--- RECOGNIZING FACE FROM IMAGE PATH: {image_path} ---")
    if not init_face_recognition_models():
        return False, None, "FACE RECOGNITION SYSTEM NOT FULLY LOADED. PLEASE CHECK SERVER LOGS."
    
    if not os.path.exists(image_path):
        print(f"ERROR: IMAGE FILE NOT FOUND AT PATH: {image_path}")
        return False, None, f"IMAGE FILE NOT FOUND AT PATH: {image_path}"

    try:
        # Read the image using OpenCV (imread reads as BGR)
        cv2_bgr_image = cv2.imread(image_path)
        if cv2_bgr_image is None:
            print(f"ERROR: FAILED TO LOAD IMAGE FROM PATH '{image_path}'. Check file corruption or permissions.")
            return False, None, "FAILED TO LOAD IMAGE FROM PATH. CHECK FILE CORRUPTION OR PERMISSIONS."
        
        # Convert BGR to RGB
        cv2_rgb_image = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)
        print(f"DEBUG RECOGNIZE_PATH: Loaded and converted image to RGB. Shape: {cv2_rgb_image.shape}")
        
        face_coords = get_largest_face_coordinates(cv2_rgb_image)
        
        if face_coords is None:
            return False, None, "NO FACE DETECTED IN THE IMAGE. PLEASE TRY AGAIN WITH A CLEARER PHOTO."

        x, y, w, h = face_coords
        
        if w <= 0 or h <= 0 or (w * h) < MIN_FACE_AREA_PIXELS:
            print(f"DEBUG RECOGNIZE_PATH: INVALID FACE ROI DIMENSIONS: W={w}, H={h}, Area={w*h}. Threshold={MIN_FACE_AREA_PIXELS}")
            return False, None, "FACE NOT CLEAR: TOO SMALL OR PARTIALLY VISIBLE."
        
        face_roi_rgb = cv2_rgb_image[y:y+h, x:x+w]
        print(f"DEBUG RECOGNIZE_PATH: Cropped face ROI. Shape: {face_roi_rgb.shape}")
        
        if face_roi_rgb.size == 0:
            return False, None, "CROPPED FACE IMAGE IS EMPTY."
        
        brightness_issue_type = detect_brightness_issue(face_roi_rgb)
        if brightness_issue_type:
            msg = ""
            if brightness_issue_type == "TOO_DARK": msg = "FACE NOT CLEAR: IMAGE TOO DARK. USE BETTER LIGHTING."
            elif brightness_issue_type == "TOO_BRIGHT": msg = "FACE NOT CLEAR: IMAGE TOO BRIGHT. REDUCE LIGHTING."
            elif brightness_issue_type == "LOW_CONTRAST": msg = "FACE NOT CLEAR: IMAGE HAS LOW CONTRAST. ADJUST LIGHTING."
            else: msg = "FACE NOT CLEAR DUE TO IMAGE QUALITY ISSUES."
            print(f"DEBUG RECOGNIZE_PATH: Image quality issue: {brightness_issue_type}. Message: {msg}")
            return False, None, msg

        current_embedding = get_facenet_embedding_from_cv2_rgb(face_roi_rgb)
        if current_embedding is None:
            return False, None, "FAILED TO GENERATE FACE EMBEDDING. PLEASE TRY AGAIN."
        print(f"DEBUG RECOGNIZE_PATH: Generated embedding of shape: {current_embedding.shape}")
        
        # Predict using the trained classifier
        if classifier is None or label_encoder is None:
            print("DEBUG RECOGNIZE_PATH: Classifier/Label Encoder not loaded, attempting to load again.")
            _load_classifier_and_label_encoder()
            if classifier is None or label_encoder is None:
                return False, None, "CLASSIFIER NOT LOADED. RUN TRAINING FIRST."

        prediction = classifier.predict([current_embedding])
        probabilities = classifier.predict_proba([current_embedding])[0]
        max_probability = np.max(probabilities)
        predicted_roll_no = label_encoder.inverse_transform(prediction)[0].upper()
        
        print(f"DEBUG RECOGNIZE_PATH: CLASSIFIER PREDICTED ROLL NO: {predicted_roll_no} (CONFIDENCE: {max_probability:.2f})")

        if max_probability < CONFIDENCE_THRESHOLD:
            print(f"DEBUG RECOGNIZE_PATH: CLASSIFIER CONFIDENCE {max_probability:.2f} IS BELOW THRESHOLD {CONFIDENCE_THRESHOLD}. REJECTED.")
            return False, None, "FACE NOT RECOGNIZED WITH SUFFICIENT CONFIDENCE. PLEASE TRY AGAIN."

        # Verify with FaceNet distance against the stored embedding for the predicted roll_no
        npy_path = os.path.join(FACE_DATA_DIR, f'{predicted_roll_no}.npy')
        print(f"DEBUG RECOGNIZE_PATH: Checking stored embedding at: {npy_path}")

        if os.path.exists(npy_path):
            try:
                stored_embedding = np.load(npy_path)
                distance = np.linalg.norm(current_embedding - stored_embedding)
                
                print(f"DEBUG RECOGNIZE_PATH: COMPARING WITH STORED EMBEDDING FOR {predicted_roll_no}. DISTANCE: {distance:.4f} (THRESHOLD: {STRICT_DISTANCE_THRESHOLD})")

                if distance < STRICT_DISTANCE_THRESHOLD:
                    return True, {'roll_no': predicted_roll_no, 'embedding': current_embedding}, f"RECOGNIZED {predicted_roll_no} (CONFIDENCE: {max_probability:.2f}, DISTANCE: {distance:.2f})"
                else:
                    print(f"DEBUG RECOGNIZE_PATH: PREDICTED {predicted_roll_no} BUT DISTANCE {distance:.4f} > {STRICT_DISTANCE_THRESHOLD}. NOT A STRONG ENOUGH MATCH.")
                    return False, None, "FACE NOT RECOGNIZED. PLEASE TRY AGAIN."
            except Exception as e:
                print(f"DEBUG RECOGNIZE_PATH: ERROR LOADING OR COMPARING EMBEDDING FOR {predicted_roll_no} from {npy_path}: {e}")
                traceback.print_exc()
                return False, None, "ERROR DURING FACE VERIFICATION. PLEASE RETRY."
        else:
            print(f"DEBUG RECOGNIZE_PATH: .NPY FILE NOT FOUND FOR PREDICTED ROLL NO {predicted_roll_no}. CANNOT PERFORM EMBEDDING DISTANCE VERIFICATION.")
            return False, None, "FACE NOT RECOGNIZED. ENROLLMENT DATA MISSING OR CORRUPT."
    except Exception as e:
        print(f"CRITICAL ERROR: GENERAL ERROR IN RECOGNIZE_FACE_FROM_IMAGE_PATH: {e}")
        traceback.print_exc()
        return False, None, f"AN UNEXPECTED ERROR OCCURRED: {e}"


def verify_face_consistency_from_image_path(image_path, required_matches=3, total_attempts=5):
    """
    Attempts to recognize a face from a saved image multiple times to verify consistency.
    Returns (True, recognized_roll_no, message) if consistency is met,
    (False, None, error_message) otherwise.
    """
    if not os.path.exists(image_path):
        return False, None, f"IMAGE FILE NOT FOUND AT PATH: {image_path}"

    recognition_counts = {}
    
    print(f"\n--- VERIFYING FACE CONSISTENCY for {image_path} (Attempts: {total_attempts}, Required Matches: {required_matches}) ---")

    for i in range(total_attempts):
        print(f"Attempt {i+1}/{total_attempts}...")
        # Use the single recognition function
        success, info, message = recognize_face_from_image_path(image_path)
        
        if success and info and 'roll_no' in info:
            roll_no = info['roll_no']
            recognition_counts[roll_no] = recognition_counts.get(roll_no, 0) + 1
            print(f"  Recognized: {roll_no}. Current count: {recognition_counts[roll_no]}")
        else:
            print(f"  Recognition failed or uncertain: {message}")
            
    verified_roll_no = None
    for roll_no, count in recognition_counts.items():
        if count >= required_matches:
            verified_roll_no = roll_no
            break

    if verified_roll_no:
        final_message = f"VERIFIED: Face consistently recognized as {verified_roll_no} in {recognition_counts[verified_roll_no]}/{total_attempts} attempts."
        print(final_message)
        return True, verified_roll_no, final_message
    else:
        final_message = "VERIFICATION FAILED: Face could not be consistently recognized or did not meet the required number of matches."
        # Provide more detail on why it failed if possible
        if recognition_counts:
            most_common = max(recognition_counts, key=recognition_counts.get)
            final_message += f" Most frequent guess: {most_common} ({recognition_counts[most_common]} times)."
        print(final_message)
        return False, None, final_message


def train_face_classifier():
    """
    Trains the SVM classifier using FaceNet embeddings from student images.
    """
    print("\n--- STARTING FACE CLASSIFIER TRAINING ---")
    # Ensure models are initialized (especially FaceNet and Haar Cascade)
    if not init_face_recognition_models():
        print("CRITICAL: Models not fully loaded for training. Aborting training.")
        return False

    faces_dataset_path = 'faces_dataset'
    if not os.path.exists(faces_dataset_path):
        print(f"ERROR: '{faces_dataset_path}' DIRECTORY NOT FOUND. CANNOT TRAIN CLASSIFIER.")
        print("Please ensure you have created this directory and placed student face images inside, categorized by folders (e.g., faces_dataset/ROLL_NO_1/img1.jpg, faces_dataset/ROLL_NO_2/img2.jpg).")
        return False

    X = [] # Embeddings
    y = [] # Labels (student roll numbers)
    
    print(f"SCANNING '{faces_dataset_path}' FOR IMAGES...")
    for person_name_folder in os.listdir(faces_dataset_path):
        person_dir = os.path.join(faces_dataset_path, person_name_folder)
        if os.path.isdir(person_dir):
            current_person_label = person_name_folder.upper() # Ensure labels are uppercase
            print(f"  Processing folder: {current_person_label}")

            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is None:
                            print(f"  WARNING: Could not read image {img_path}. Skipping.")
                            continue
                        
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # Detect face and get ROI
                        face_coords = get_largest_face_coordinates(img_rgb)
                        if face_coords is None:
                            print(f"  WARNING: NO FACE DETECTED in {img_path}. SKIPPING.")
                            continue
                        
                        x, y, w, h = face_coords
                        face_roi_rgb = img_rgb[y:y+h, x:x+w]

                        # Generate embedding
                        embedding = get_facenet_embedding_from_cv2_rgb(face_roi_rgb)
                        if embedding is not None:
                            X.append(embedding)
                            y.append(current_person_label) # Assign uppercase label
                            print(f"    - Added embedding for {img_name}")
                        else:
                            print(f"  WARNING: FAILED TO GET EMBEDDING for {img_path}. SKIPPING.")
                    except Exception as e:
                        print(f"  ERROR PROCESSING IMAGE {img_path}: {e}")
                        traceback.print_exc()

    if len(X) == 0:
        print("ERROR: NO VALID FACE IMAGES FOUND OR PROCESSED FOR TRAINING. Training aborted.")
        return False

    print(f"\nGenerated {len(X)} embeddings for training from {len(np.unique(y))} unique individuals.")

    # Encode labels
    global label_encoder
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y)
        print(f"Label classes: {label_encoder.classes_}")
    except ValueError as e:
        print(f"ERROR: Could not encode labels. This might happen if all images belong to the same person. Need at least 2 unique persons for classification. Details: {e}")
        return False


    # Train SVM classifier
    print("TRAINING SVM CLASSIFIER...")
    
    # For simplicity, a basic SVM. Consider GridSearchCV for better performance.
    global classifier
    classifier = SVC(kernel='linear', probability=True)
    try:
        classifier.fit(X, y_encoded)
        print("SVM CLASSIFIER TRAINING COMPLETE.")
    except Exception as e:
        print(f"ERROR: Failed to train SVM classifier: {e}")
        traceback.print_exc()
        return False

    # Save the trained classifier and label encoder
    try:
        joblib.dump(classifier, CLASSIFIER_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        print(f"CLASSIFIER SAVED TO '{CLASSIFIER_PATH}'")
        print(f"LABEL ENCODER SAVED TO '{LABEL_ENCODER_PATH}'")
        return True
    except Exception as e:
        print(f"ERROR SAVING CLASSIFIER OR LABEL ENCODER: {e}")
        traceback.print_exc()
        return False

def run_realtime_face_recognition():
    """
    Performs real-time face recognition using webcam feed.
    """
    print("\n--- STARTING REAL-TIME FACE RECOGNITION (WEBCAM) ---")
    if not init_face_recognition_models():
        print("REQUIRED MODELS/CLASSIFIERS ARE NOT LOADED. ABORTING REAL-TIME RECOGNITION.")
        print("Please ensure all models ('facenet_pytorch', 'haarcascade_frontalface_default.xml') are loadable")
        print("and 'face_classifier.pkl', 'label_encoder.pkl' exist (run train_face_classifier() first).")
        return

    cap = cv2.VideoCapture(0) # 0 is typically the default webcam

    if not cap.isOpened():
        print("ERROR: COULD NOT OPEN WEBCAM. PLEASE CHECK IF IT'S CONNECTED AND NOT IN USE.")
        return

    print("Press 'Q' to quit the webcam feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("FAILED TO GRAB FRAME. EXITING WEBCAM FEED...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_coords = get_largest_face_coordinates(rgb_frame)

        text_display = "NO FACE DETECTED"
        color_display = (0, 0, 255) # Red for unknown/no face

        if faces_coords is not None:
            x, y, w, h = faces_coords
            
            # Check face dimensions after detection (redundant with previous check but good for safety)
            if w <= 0 or h <= 0 or (w * h) < MIN_FACE_AREA_PIXELS:
                text_display = "FACE TOO SMALL/INVALID"
                color_display = (0, 0, 255) # Red
            else:
                face_roi_rgb = rgb_frame[y:y+h, x:x+w]

                brightness_issue = detect_brightness_issue(face_roi_rgb)
                if brightness_issue:
                    text_display = f"IMAGE ISSUE: {brightness_issue.replace('_', ' ')}"
                    color_display = (0, 165, 255) # Orange
                else:
                    try:
                        current_embedding = get_facenet_embedding_from_cv2_rgb(face_roi_rgb)

                        if current_embedding is not None:
                            prediction = classifier.predict([current_embedding])
                            probabilities = classifier.predict_proba([current_embedding])[0]
                            max_probability = np.max(probabilities)
                            predicted_label_classifier = label_encoder.inverse_transform(prediction)[0].upper()

                            if max_probability >= CONFIDENCE_THRESHOLD:
                                npy_path = os.path.join(FACE_DATA_DIR, f'{predicted_label_classifier}.npy')
                                if os.path.exists(npy_path):
                                    stored_embedding = np.load(npy_path)
                                    distance = np.linalg.norm(current_embedding - stored_embedding)

                                    if distance < STRICT_DISTANCE_THRESHOLD:
                                        text_display = f"{predicted_label_classifier} ({max_probability:.2f})"
                                        color_display = (0, 255, 0) # Green
                                    else:
                                        text_display = f"UNKNOWN (DIST: {distance:.2f})"
                                        color_display = (0, 0, 255) # Red (distance too high)
                                else:
                                    text_display = "UNKNOWN (NPY MISSING)"
                                    color_display = (0, 0, 255) # Red (no NPY file for prediction)
                            else:
                                text_display = f"UNKNOWN ({max_probability:.2f})"
                                color_display = (0, 0, 255) # Red (confidence too low)
                        else:
                            text_display = "EMBEDDING FAILED"
                            color_display = (0, 0, 255)
                    except Exception as e:
                        text_display = "PROCESSING ERROR"
                        color_display = (0, 0, 255) # Red
                        print(f"ERROR DURING REAL-TIME FACE RECOGNITION: {e}")
                        traceback.print_exc()

            cv2.rectangle(frame, (x, y), (x+w, y+h), color_display, 2)
            cv2.putText(frame, text_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_display, 2)

        cv2.imshow('REAL-TIME FACE RECOGNITION (Press Q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time recognition session ended.")

# Initialize models when this module is imported
# This ensures models are loaded when the service is used by Flask or directly
init_face_recognition_models()

# Example usage for new functions (can be uncommented for testing purposes ONLY when running face_recognition_service.py directly)
if __name__ == '__main__':
    print("\n--- Running face_recognition_service.py directly for testing purposes ---")
    print("This script will attempt to train the classifier and then run real-time recognition.")
    print("Ensure you have a 'faces_dataset' directory with student subfolders for training.")

    # Try training the classifier
    training_success = train_face_classifier()
    if training_success:
        print("\nClassifier training completed. Proceeding to real-time recognition demo (if webcam is available).")
        # You can add a prompt here to ask if the user wants to run real-time recognition
        # For now, it will proceed if training was successful.
        run_realtime_face_recognition()
    else:
        print("\nClassifier training failed. Real-time recognition demo skipped.")

    # --- Additional specific tests (uncomment to run) ---
    # print("\n--- DEMONSTRATING SINGLE IMAGE RECOGNITION ---")
    # # Replace with an actual path to a registered student's image for testing
    # test_image_path = 'static/student_images/YOUR_ROLL_NO.PNG' # E.g., 'static/student_images/JOHN_DOE.PNG'
    # if os.path.exists(test_image_path):
    #     success, info, message = recognize_face_from_image_path(test_image_path)
    #     if success:
    #         print(f"Single image recognition result: SUCCESS! Recognized as {info['roll_no']}. Message: {message}")
    #     else:
    #         print(f"Single image recognition result: FAILED! Message: {message}")
    # else:
    #     print(f"SKIPPING single image recognition test: Test image '{test_image_path}' not found.")

    # print("\n--- DEMONSTRATING CONSISTENCY VERIFICATION ---")
    # # Replace with an actual path to a registered student's image for testing
    # test_image_path_for_consistency = 'static/student_images/YOUR_ROLL_NO.PNG'
    # if os.path.exists(test_image_path_for_consistency):
    #     verified, roll_no, verification_message = verify_face_consistency_from_image_path(
    #         test_image_path_for_consistency,
    #         required_matches=3,
    #         total_attempts=5
    #     )
    #     if verified:
    #         print(f"Consistency verification result: VERIFIED! As {roll_no}. Message: {verification_message}")
    #     else:
    #         print(f"Consistency verification result: FAILED! Message: {verification_message}")
    # else:
    #     print(f"SKIPPING consistency verification test: Test image '{test_image_path_for_consistency}' not found.")

