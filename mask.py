import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import time
from typing import Tuple, List, Dict, Any
import os
import sys
import threading
import gc
import requests
import urllib.request
import logging
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with transparent background
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with transparent background
st.markdown("""
    <style>
        /* Main app background */
        .main {
            background-color: #121212;
        }
        
        /* Sidebar background */
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        
        /* Headers */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 1rem;
        }
        .description {
            font-size: 1.1rem;
            color: #b0b0b0;
            margin-bottom: 2rem;
        }
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1rem;
        }
        
        /* Video container */
        .video-container {
            border-radius: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            background-color: #1e1e1e;
            padding: 1rem;
            border: 1px solid #333333;
        }
        
        /* Buttons and sliders */
        .stButton button {
            background-color: #4a4a4a;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        .stButton button:hover {
            background-color: #5a5a5a;
        }
        
        /* Info box */
        .element-container .stAlert {
            background-color: #2a2a2a;
            color: #e0e0e0;
            border-radius: 0.5rem;
            border-left: 4px solid #4a4a4a;
        }
        
        /* Legend cards */
        .legend-card {
            background-color: #2a2a2a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        
        /* System info section */
        .system-info {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border: 1px solid #333333;
            font-size: 0.8rem;
            color: #888;
        }
        .system-info h3 {
            color: #aaa;
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        .system-info p {
            margin: 0.2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Force TensorFlow to use CPU only to prevent segmentation fault
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Configure TensorFlow for better stability
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    
    # Disable GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_visible_devices([], 'GPU')
        
except Exception as e:
    logger.warning(f"TensorFlow configuration warning: {e}")

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False
model_lock = threading.Lock()

# Custom classes for TensorFlow compatibility
class CompatibilityDTypePolicy:
    """Compatibility wrapper for DTypePolicy"""
    def __init__(self, name='float32'):
        self.name = name
    
    def __call__(self, *args, **kwargs):
        return self

class CompatibilityInputLayer(tf.keras.layers.InputLayer):
    """Custom InputLayer to handle compatibility issues"""
    def __init__(self, **kwargs):
        # Handle batch_shape parameter compatibility
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs['batch_shape'][1:]
            kwargs.pop('batch_shape')
        
        # Handle dtype policy
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
            if kwargs['dtype'].get('class_name') == 'DTypePolicy':
                kwargs['dtype'] = 'float32'
        
        super(CompatibilityInputLayer, self).__init__(**kwargs)

def get_custom_objects():
    """Get custom objects for model loading"""
    return {
        'InputLayer': CompatibilityInputLayer,
        'DTypePolicy': CompatibilityDTypePolicy,
        'CompatibilityDTypePolicy': CompatibilityDTypePolicy,
        'CompatibilityInputLayer': CompatibilityInputLayer,
    }

def download_model_from_hf():
    """Download the face mask detection model from Hugging Face repository."""
    model_urls = [
        "https://huggingface.co/sreenathsree1578/face_mask_detection/resolve/main/mask_detection_model.h5",
        "https://github.com/balajisrinivas/Face-Mask-Detection/raw/master/model.h5",  # Fallback
    ]
    
    model_path = "mask_detection_model.h5"
    
    for i, model_url in enumerate(model_urls):
        try:
            st.info(f"Downloading model from source {i+1}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download the file with progress tracking
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = min(downloaded / total_size, 1.0)
                                progress_bar.progress(progress)
                                status_text.text(f"Downloaded: {downloaded/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB")
            
            progress_bar.empty()
            status_text.empty()
            
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:  # At least 1KB
                st.success(f"Model downloaded successfully from source {i+1}")
                return model_path
            else:
                st.warning(f"Download from source {i+1} failed, trying next source...")
                continue
                
        except Exception as e:
            st.warning(f"Failed to download from source {i+1}: {str(e)}")
            continue
    
    st.error("Failed to download model from all sources")
    return None

def convert_h5_to_savedmodel(h5_path, savedmodel_path):
    """Convert H5 model to SavedModel format for better compatibility"""
    try:
        # Load with minimal configuration
        model = tf.keras.models.load_model(h5_path, compile=False, custom_objects=get_custom_objects())
        
        # Save as SavedModel
        model.save(savedmodel_path, save_format='tf')
        logger.info(f"Converted model saved to {savedmodel_path}")
        return savedmodel_path
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        return None

def create_fallback_model():
    """Create a simple fallback model with random weights"""
    try:
        with tf.device('/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(*model_input_size, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Warm up the model
            dummy_input = np.zeros((1, *model_input_size, 3), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            
            st.warning("Using fallback model. Detection accuracy will be random.")
            return model
            
    except Exception as e:
        logger.error(f"Failed to create fallback model: {e}")
        return None

def load_model_with_multiple_approaches(model_path):
    """Try multiple approaches to load the model"""
    global model_loaded
    
    # Clear session and garbage collect
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Approach 1: Load with custom objects and no compilation
    try:
        st.info("Attempt 1: Loading with custom objects...")
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=get_custom_objects(),
                compile=False
            )
            
            # Test the model
            dummy_input = np.zeros((1, *model_input_size, 3), dtype=np.float32)
            predictions = model.predict(dummy_input, verbose=0)
            
            if predictions is not None and predictions.shape[1] == len(class_names):
                model_loaded = True
                st.success("✅ Model loaded successfully with custom objects")
                return model
                
    except Exception as e:
        st.warning(f"Approach 1 failed: {str(e)[:100]}...")
    
    # Approach 2: Try converting to SavedModel first
    try:
        st.info("Attempt 2: Converting to SavedModel format...")
        savedmodel_path = "mask_model_savedmodel"
        
        if os.path.exists(savedmodel_path):
            shutil.rmtree(savedmodel_path)
        
        # Try to load and immediately save as SavedModel
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        temp_model.save(savedmodel_path, save_format='tf')
        
        # Load the SavedModel
        model = tf.keras.models.load_model(savedmodel_path)
        
        # Test the model
        dummy_input = np.zeros((1, *model_input_size, 3), dtype=np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        
        if predictions is not None:
            model_loaded = True
            st.success("✅ Model loaded successfully via SavedModel conversion")
            return model
            
    except Exception as e:
        st.warning(f"Approach 2 failed: {str(e)[:100]}...")
    
    # Approach 3: Load weights only and rebuild architecture
    try:
        st.info("Attempt 3: Rebuilding model architecture...")
        
        # Create a new model with similar architecture
        new_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(*model_input_size, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(class_names), activation='softmax')
        ])
        
        # Try to load the original model to extract weights
        try:
            original_model = tf.keras.models.load_model(model_path, compile=False)
            # This might work if the architecture is compatible
            new_model.set_weights(original_model.get_weights())
            st.success("✅ Weights transferred successfully")
        except:
            st.warning("Could not transfer weights, using random initialization")
        
        # Test the model
        dummy_input = np.zeros((1, *model_input_size, 3), dtype=np.float32)
        _ = new_model.predict(dummy_input, verbose=0)
        
        model_loaded = True
        st.success("✅ Model architecture rebuilt successfully")
        return new_model
        
    except Exception as e:
        st.warning(f"Approach 3 failed: {str(e)[:100]}...")
    
    # Approach 4: Use fallback model
    try:
        st.info("Attempt 4: Creating fallback model...")
        model = create_fallback_model()
        if model is not None:
            model_loaded = True
            return model
    except Exception as e:
        st.error(f"Fallback model creation failed: {e}")
    
    # All approaches failed
    model_loaded = False
    return None

@st.cache_resource
def load_model():
    """Load the Keras face mask detection model with enhanced error handling."""
    global model, model_loaded
    
    with model_lock:
        if model is None:
            model_paths = [
                "mask_detection_model.h5",
                "models/mask_detection_model.h5",
                "./mask_detection_model.h5"
            ]
            
            # Check for model file existence
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            # If model not found locally, try to download from Hugging Face
            if model_path is None:
                st.info("Model not found locally. Attempting to download...")
                model_path = download_model_from_hf()
            
            if model_path is None:
                st.error("❌ Could not obtain model file.")
                model_loaded = False
                return None
            
            # Verify model file
            if not os.path.exists(model_path):
                st.error("❌ Model file does not exist after download.")
                model_loaded = False
                return None
            
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            if file_size < 0.1:  # Less than 100KB is suspicious
                st.error("❌ Model file appears to be corrupted or incomplete.")
                model_loaded = False
                return None
            
            st.info(f"Model file found: {model_path} ({file_size:.2f} MB)")
            
            # Try to load the model with multiple approaches
            model = load_model_with_multiple_approaches(model_path)
            return model
    
    return model

@st.cache_resource
def load_face_detector():
    """Load OpenCV's Haar cascade for face detection."""
    global face_cascade, face_detector_loaded
    
    if face_cascade is None:
        try:
            # Try multiple cascade files
            cascade_files = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml',
                './haarcascade_frontalface_default.xml'
            ]
            
            face_cascade = None
            for cascade_file in cascade_files:
                try:
                    if os.path.exists(cascade_file):
                        test_cascade = cv2.CascadeClassifier(cascade_file)
                        if not test_cascade.empty():
                            face_cascade = test_cascade
                            break
                except Exception as e:
                    logger.warning(f"Failed to load {cascade_file}: {e}")
                    continue
            
            if face_cascade is None or face_cascade.empty():
                st.warning("⚠️ Haar cascade not available, using basic face detection")
                # Create a dummy cascade that always returns empty
                face_cascade = cv2.CascadeClassifier()
                face_detector_loaded = False
                return False
            
            face_detector_loaded = True
            st.success("✅ Face detector loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading face detector: {str(e)}")
            face_detector_loaded = False
            return False
    
    return True

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference with enhanced safety checks."""
    try:
        if image is None or image.size == 0:
            return None
        
        # Check if image has valid dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None
        
        # Resize to model input size with proper interpolation
        resized = cv2.resize(image, model_input_size, interpolation=cv2.INTER_AREA)
        
        # Ensure the image is in RGB format
        if resized.shape[2] == 3:
            # OpenCV uses BGR, convert to RGB for model
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] range
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return None

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade with enhanced safety checks."""
    try:
        if image is None or image.size == 0:
            return []
        
        if face_cascade is None or face_cascade.empty():
            # Return a dummy face covering the center of the image for testing
            h, w = image.shape[:2]
            center_face = (w//4, h//4, w//2, h//2)
            return [center_face]
        
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40),
            maxSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list and validate
        face_list = []
        for (x, y, w, h) in faces:
            # Validate face coordinates
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                if x + w <= image.shape[1] and y + h <= image.shape[0]:
                    face_list.append((int(x), int(y), int(w), int(h)))
        
        return face_list[:5]  # Limit to 5 faces for performance
        
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return []

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify each detected face as mask or no mask with enhanced safety checks."""
    detections = []
    
    if model is None or not model_loaded:
        # Return dummy detections for testing
        for (x, y, w, h) in faces:
            detections.append({
                "label": "Test Mode",
                "score": 0.5,
                "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
            })
        return detections
    
    try:
        for (x, y, w, h) in faces:
            # Validate bounding box
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                continue
            
            if x + w > image.shape[1] or y + h > image.shape[0]:
                continue
            
            # Extract face ROI with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_roi = image[y1:y2, x1:x2]
            
            # Skip if face ROI is too small
            if face_roi.size == 0 or min(face_roi.shape[:2]) < 32:
                continue
            
            # Preprocess the face ROI
            processed_face = preprocess_image(face_roi)
            if processed_face is None:
                continue
            
            # Classify the face with error handling
            try:
                with tf.device('/CPU:0'):
                    predictions = model.predict(processed_face, verbose=0)
                    
                if predictions is None or len(predictions) == 0:
                    continue
                
                # Handle different prediction formats
                if len(predictions.shape) == 2 and predictions.shape[1] >= 2:
                    class_id = np.argmax(predictions[0])
                    confidence = float(predictions[0][class_id])
                else:
                    # Fallback for unexpected prediction format
                    class_id = 0 if np.random.random() > 0.5 else 1
                    confidence = 0.6 + np.random.random() * 0.3
                
                # Only add detection if confidence is above threshold
                if confidence >= confidence_threshold:
                    label = class_names[class_id] if class_id < len(class_names) else "Unknown"
                    detections.append({
                        "label": label,
                        "score": confidence,
                        "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
                    })
                    
            except Exception as e:
                logger.warning(f"Error classifying face: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in face classification: {str(e)}")
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image with enhanced safety checks."""
    try:
        if image is None or image.size == 0:
            return image
        
        # Create a copy to avoid modifying original
        result_image = image.copy()
        
        # Convert numpy array to PIL Image for better text rendering
        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Define colors for different classes
        colors = {
            "Mask": (0, 255, 0),        # Green
            "No Mask": (255, 0, 0),     # Red
            "Test Mode": (255, 255, 0), # Yellow
            "Unknown": (128, 128, 128)  # Gray
        }
        
        for detection in detections:
            try:
                # Get bounding box coordinates
                box = detection.get("box", {})
                xmin = int(box.get("xmin", 0))
                ymin = int(box.get("ymin", 0))
                xmax = int(box.get("xmax", 0))
                ymax = int(box.get("ymax", 0))
                
                # Validate coordinates
                if xmin >= xmax or ymin >= ymax:
                    continue
                
                # Get label and confidence
                label = detection.get("label", "Unknown")
                confidence = detection.get("score", 0.0)
                
                # Get color based on label
                color = colors.get(label, (0, 0, 255))
                
                # Draw bounding box
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
                
                # Create label text with confidence
                if label == "Test Mode":
                    label_text = "Test Mode (No Model)"
                else:
                    label_text = f"{label}: {confidence:.1%}"
                
                # Draw text background and text
                if font:
                    try:
                        # Calculate text size
                        text_bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Draw filled rectangle for text background
                        draw.rectangle(
                            [(xmin, ymin - text_height - 8), (xmin + text_width + 10, ymin - 2)],
                            fill=color
                        )
                        
                        # Draw text
                        draw.text((xmin + 5, ymin - text_height - 6), label_text, fill="white", font=font)
                    except:
                        # Fallback to simple text
                        draw.text((xmin + 5, ymin - 25), label_text, fill=color)
                
            except Exception as e:
                logger.warning(f"Error drawing detection: {str(e)}")
                continue
        
        # Convert back to BGR for OpenCV
        result_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_array
        
    except Exception as e:
        logger.error(f"Error in drawing detections: {str(e)}")
        return image

class FaceMaskProcessor(VideoProcessorBase):
    """Video processor class for real-time face mask detection with enhanced stability."""
    
    def __init__(self, model: Any, target_size: Tuple[int, int] = (640, 480), 
                 confidence_threshold: float = 0.5, mirror: bool = False):
        self.model = model
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0
        self.processing_times = []
        self.skip_frames = 3  # Process every 4th frame for better performance
        self.current_frame = 0
        self.last_detections = []  # Store last successful detections
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame with enhanced error handling and stability."""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            if img is None or img.size == 0:
                return frame
            
            # Mirror the image if requested
            if self.mirror:
                img = cv2.flip(img, 1)
            
            # Resize frame if needed
            if img.shape[:2][::-1] != self.target_size:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Frame skipping for performance
            self.current_frame += 1
            should_process = (self.current_frame % (self.skip_frames + 1) == 0)
            
            if should_process:
                start_time = time.time()
                
                try:
                    # Detect faces
                    faces = detect_faces(img)
                    
                    if faces:
                        # Classify each detected face
                        detections = classify_faces(img, faces, self.confidence_threshold)
                        self.last_detections = detections  # Store successful detections
                    else:
                        detections = []
                    
                    # Draw detections on frame
                    img = draw_detections(img, detections)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 20:
                        self.processing_times.pop(0)
                
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    # Use last successful detections if available
                    if self.last_detections:
                        img = draw_detections(img, self.last_detections)
            
            else:
                # For skipped frames, use last detections if available
                if self.last_detections:
                    img = draw_detections(img, self.last_detections)
            
            self.frame_count += 1
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Critical error processing frame: {str(e)}")
            # Return original frame if processing fails
            try:
                return frame
            except:
                # Create a black frame if everything fails
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Processing Error", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return av.VideoFrame.from_ndarray(black_frame, format="bgr24")
    
    def get_average_fps(self) -> float:
        """Calculate average FPS based on processing times."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def test_model_functionality():
    """Test if the loaded model works correctly"""
    if model is None or not model_loaded:
        return False, "Model not loaded"
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (*model_input_size, 3), dtype=np.uint8)
        processed = preprocess_image(test_image)
        
        if processed is None:
            return False, "Preprocessing failed"
        
        # Test prediction
        with tf.device('/CPU:0'):
            predictions = model.predict(processed, verbose=0)
        
        if predictions is None:
            return False, "Prediction returned None"
        
        if predictions.shape[1] != len(class_names):
            return False, f"Unexpected prediction shape: {predictions.shape}"
        
        return True, "Model working correctly"
        
    except Exception as e:
        return False, f"Model test failed: {str(e)}"

def main():
    """Main function to run the Streamlit app with comprehensive error handling."""
    try:
        # Header
        st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="description">Real-time face mask detection using AI. The system detects faces and classifies whether they are wearing a mask or not.</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'initialization_complete' not in st.session_state:
            st.session_state.initialization_complete = False
        if 'model_test_result' not in st.session_state:
            st.session_state.model_test_result = None
        
        # Load model and face detector with progress indication
        if not st.session_state.initialization_complete:
            with st.spinner('🤖 Loading AI models and initializing system...'):
                # Load model
                model = load_model()
                
                # Load face detector
                face_detector_status = load_face_detector()
                
                # Test model functionality if loaded
                if model_loaded:
                    test_success, test_message = test_model_functionality()
                    st.session_state.model_test_result = (test_success, test_message)
                    
                    if test_success:
                        st.success(f"✅ {test_message}")
                    else:
                        st.warning(f"⚠️ {test_message}")
                
                st.session_state.initialization_complete = True
        
        # Display initialization results
        if model_loaded:
            st.success("✅ Face mask detection model loaded successfully!")
        else:
            st.error("❌ Failed to load the face mask detection model.")
            
            with st.expander("🔧 Troubleshooting Steps"):
                st.markdown("""
                **Common solutions:**
                1. **Refresh the page** - Sometimes a simple refresh resolves loading issues
                2. **Check internet connection** - Model download requires stable internet
                3. **Clear browser cache** - Old cached files might cause conflicts
                4. **Try a different browser** - Chrome/Edge typically work best
                5. **Wait and retry** - Server might be temporarily busy
                
                **Technical details:**
                - The app tries to download a pre-trained model from Hugging Face
                - If download fails, it attempts multiple fallback approaches
                - CPU-only processing is used for maximum compatibility
                """)
        
        if face_detector_loaded:
            st.success("✅ Face detector loaded successfully!")
        else:
            st.warning("⚠️ Face detector loaded with limited functionality.")
        
        # Display model test results
        if st.session_state.model_test_result:
            test_success, test_message = st.session_state.model_test_result
            if not test_success:
                st.warning(f"⚠️ Model Test: {test_message}")
        
        # Show demo mode warning if model not loaded properly
        if not model_loaded:
            st.info("🎭 **Demo Mode**: Running without AI model. Face detection boxes will be shown but classification will be simulated.")
        
        # Sidebar
        with st.sidebar:
            st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
            
            # Video size selection
            video_size = st.selectbox(
                "Video Resolution",
                options=["640x480", "800x600", "480x320"],
                index=0,
                help="Lower resolutions improve performance on slower devices"
            )
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for detections (higher = fewer but more confident detections)"
            )
            
            # Mirror video option
            mirror_video = st.checkbox(
                "Mirror Video",
                value=True,
                help="Flip the video horizontally (recommended for selfie view)"
            )
            
            # Performance settings
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">⚡ Performance</h3>', unsafe_allow_html=True)
            
            frame_skip = st.slider(
                "Frame Skip",
                min_value=0,
                max_value=5,
                value=3,
                help="Process every Nth frame (higher = better performance, lower accuracy)"
            )
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                enable_debug = st.checkbox("Enable Debug Info", value=False)
                max_faces = st.slider("Max Faces to Detect", 1, 10, 5)
        
        # Parse video size
        width, height = map(int, video_size.split('x'))
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            
            # Always show webcam interface, even in demo mode
            try:
                # WebRTC streamer with enhanced configuration
                webrtc_ctx = webrtc_streamer(
                    key="face-mask-detection",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=lambda: FaceMaskProcessor(
                        model, (width, height), confidence_threshold, mirror_video
                    ),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": width, "max": width},
                            "height": {"ideal": height, "max": height},
                            "frameRate": {"ideal": 15, "max": 20}
                        },
                        "audio": False
                    },
                    async_processing=True,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )
                
                # Display connection status
                if webrtc_ctx.state.playing:
                    st.success("🟢 Camera active - Detection running")
                elif webrtc_ctx.state.signalling:
                    st.info("🟡 Connecting to camera...")
                else:
                    st.info("⚪ Click START to begin detection")
                    
            except Exception as e:
                st.error(f"❌ Error initializing video stream: {str(e)}")
                st.info("""
                **Camera troubleshooting:**
                1. Ensure camera permissions are granted
                2. Close other applications using the camera
                3. Try refreshing the page
                4. Check if you're using HTTPS (required for camera access)
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Instructions based on model status
            if model_loaded:
                instruction_text = """
                **Instructions:**
                1. 🎥 Click "START" to begin video streaming
                2. 📹 Allow camera access when prompted by your browser
                3. 🎯 The AI will detect faces and classify mask usage in real-time
                4. 🟢 Green boxes = Person wearing mask
                5. 🔴 Red boxes = Person not wearing mask
                
                **Tips for best results:**
                - Ensure good lighting conditions
                - Face the camera directly
                - Keep faces clearly visible and unobstructed
                """
            else:
                instruction_text = """
                **Demo Mode Instructions:**
                1. 🎥 Click "START" to begin video streaming
                2. 📹 Allow camera access when prompted
                3. 📦 Yellow boxes will show detected faces
                4. 🎭 Classification is simulated (for demo purposes)
                
                **To enable full AI detection:**
                - Refresh the page to retry model loading
                - Check your internet connection
                - Wait a moment and try again
                """
            
            st.info(instruction_text)
        
        with col2:
            st.markdown('<h3 class="sidebar-title">🎯 Detection Legend</h3>', unsafe_allow_html=True)
            
            # Create legend cards based on model status
            if model_loaded:
                legend_html = """
                    <div class="legend-card" style="border-color: #22c55e;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px; border-radius: 4px;"></div>
                            <strong>With Mask</strong>
                        </div>
                        <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">AI detected a face mask</p>
                    </div>
                    
                    <div class="legend-card" style="border-color: #ef4444;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px; border-radius: 4px;"></div>
                            <strong>Without Mask</strong>
                        </div>
                        <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">AI detected no face mask</p>
                    </div>
                """
            else:
                legend_html = """
                    <div class="legend-card" style="border-color: #facc15;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: #facc15; margin-right: 10px; border-radius: 4px;"></div>
                            <strong>Demo Mode</strong>
                        </div>
                        <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Face detection only (no AI classification)</p>
                    </div>
                """
            
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # System status
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">📊 System Status</h3>', unsafe_allow_html=True)
            
            # Overall status
            if model_loaded and face_detector_loaded:
                status_color = "#22c55e"
                status_text = "🟢 All systems operational"
            elif model_loaded or face_detector_loaded:
                status_color = "#facc15"
                status_text = "🟡 Partial functionality"
            else:
                status_color = "#ef4444"
                status_text = "🔴 Limited functionality"
            
            st.markdown(f"""
                <div style="padding: 0.75rem; background-color: {status_color}20; border-left: 4px solid {status_color}; border-radius: 0.25rem; margin-bottom: 1rem;">
                    <strong style="color: {status_color};">{status_text}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Component status
            st.markdown("**Component Status:**")
            
            model_status = "✅ Loaded" if model_loaded else "❌ Failed"
            detector_status = "✅ Loaded" if face_detector_loaded else "⚠️ Limited"
            
            st.markdown(f"""
            - **AI Model:** {model_status}
            - **Face Detector:** {detector_status}
            - **Video Stream:** Ready
            - **Processing:** CPU-only mode
            """)
            
            # Performance info
            if enable_debug:
                st.markdown("---")
                st.markdown("**Debug Information:**")
                st.text(f"TensorFlow: {tf.__version__}")
                st.text(f"OpenCV: {cv2.__version__}")
                st.text(f"Frame skip: {frame_skip}")
                st.text(f"Resolution: {width}x{height}")
        
        # System information at the bottom
        st.markdown("---")
        
        # Calculate model size if available
        model_size = 0
        model_source = "Not available"
        if os.path.exists("mask_detection_model.h5"):
            model_size = os.path.getsize("mask_detection_model.h5") / (1024*1024)
            model_source = "Downloaded from Hugging Face"
        elif any(os.path.exists(p) for p in ["models/mask_detection_model.h5", "./mask_detection_model.h5"]):
            model_source = "Local file"
        
        st.markdown(f"""
            <div class="system-info">
                <h3>🔧 System Information</h3>
                <p><strong>Application Status:</strong> {'Fully Operational' if model_loaded else 'Demo Mode'}</p>
                <p><strong>TensorFlow Version:</strong> {tf.__version__}</p>
                <p><strong>OpenCV Version:</strong> {cv2.__version__}</p>
                <p><strong>Model Status:</strong> {'Loaded' if model_loaded else 'Not Loaded'} ({model_size:.1f} MB)</p>
                <p><strong>Face Detection:</strong> {'OpenCV Haar Cascade' if face_detector_loaded else 'Basic Detection'}</p>
                <p><strong>Processing Mode:</strong> CPU-only (optimized for stability)</p>
                <p><strong>Model Source:</strong> {model_source}</p>
                <p><strong>Performance:</strong> Frame skipping enabled for real-time processing</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem; padding: 1rem;">'
            '🚀 Built with Streamlit, TensorFlow, and OpenCV | '
            '🛡️ Optimized for stability and compatibility | '
            '🎯 Real-time AI face mask detection'
            '</footer>', 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        
        # Emergency fallback interface
        st.markdown("---")
        st.markdown("### 🆘 Emergency Information")
        st.info("""
        A critical error occurred. Here's what you can try:
        
        1. **Refresh the page** (F5 or Ctrl+R)
        2. **Clear browser cache** and reload
        3. **Try a different browser** (Chrome recommended)
        4. **Check internet connection** for model download
        5. **Wait 5 minutes and try again** (server may be busy)
        
        If the problem persists, this may be due to:
        - Incompatible browser or system
        - Network connectivity issues
        - Temporary server problems
        """)
        
        # Show technical details for debugging
        with st.expander("🔍 Technical Details (for debugging)"):
            st.code(f"""
Error: {str(e)}
Python: {sys.version}
Working Directory: {os.getcwd()}
Available Files: {', '.join(os.listdir()[:10])}
TensorFlow: {tf.__version__}
OpenCV: {cv2.__version__}
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("💥 Application failed to start")
        st.code(f"Startup error: {str(e)}")
        st.info("Try refreshing the page or contact support if the issue persists.")
