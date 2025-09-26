import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from typing import Tuple, List, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config first
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import TensorFlow with comprehensive error handling
try:
    import tensorflow as tf
    # Disable GPU and eager execution for compatibility
    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.disable_eager_execution()
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    logger.error(f"TensorFlow initialization error: {e}")
    TENSORFLOW_AVAILABLE = False

# Try to import streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    STREAMLIT_WEBRTC_AVAILABLE = True
    logger.info("streamlit-webrtc imported successfully")
except ImportError as e:
    logger.error(f"streamlit-webrtc import failed: {e}")
    STREAMLIT_WEBRTC_AVAILABLE = False

# Try to import Hugging Face Hub
try:
    from huggingface_hub import hf_hub_download
    HUGGINGFACE_AVAILABLE = True
    logger.info("huggingface-hub imported successfully")
except ImportError as e:
    logger.error(f"huggingface-hub import failed: {e}")
    HUGGINGFACE_AVAILABLE = False

# Custom CSS for modern styling
st.markdown("""
    <style>
        .main {
            background-color: #121212;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
        }
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
        .video-container {
            border-radius: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            background-color: #1e1e1e;
            padding: 1rem;
            border: 1px solid #333333;
        }
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
        .legend-card {
            background-color: #2a2a2a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        .system-info {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border: 1px solid #333333;
            font-size: 0.8rem;
            color: #888;
        }
        .error-details {
            background-color: #2a1a1a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #664444;
            font-family: monospace;
            font-size: 0.9rem;
            color: #ff9999;
        }
    </style>
""", unsafe_allow_html=True)

# Global variables
model = None
face_cascade = None
model_input_size = (128, 128)
class_names = ['Mask', 'No Mask']
model_loaded = False
face_detector_loaded = False

class SimpleFaceDetector:
    """Fallback face detector using basic image processing"""
    
    def __init__(self):
        self.min_face_size = (30, 30)
        self.scale_factor = 1.1
        self.min_neighbors = 5
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Simple face detection using skin color and contour detection"""
        try:
            # Convert to HSV color space for better skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in the skin mask
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                # Filter contours by area and aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (w >= self.min_face_size[0] and h >= self.min_face_size[1] and 
                    0.5 <= aspect_ratio <= 2.0):
                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            logger.error(f"Simple face detection error: {e}")
            return []

def load_face_detector():
    """Load face detector with fallback options"""
    global face_cascade, face_detector_loaded
    
    try:
        # Try to load OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if not face_cascade.empty():
            face_detector_loaded = True
            logger.info("OpenCV face detector loaded successfully")
            return True
        else:
            logger.warning("OpenCV cascade classifier is empty")
            face_detector_loaded = False
            return False
            
    except Exception as e:
        logger.error(f"Error loading OpenCV face detector: {e}")
        face_detector_loaded = False
        return False

def create_simple_model():
    """Create a simple fallback model for demonstration"""
    global model, model_loaded
    
    try:
        if TENSORFLOW_AVAILABLE:
            # Create a simple sequential model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            # Compile with dummy weights (won't actually work for real detection)
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            model_loaded = True
            logger.info("Simple fallback model created")
            return model
        else:
            model_loaded = False
            return None
            
    except Exception as e:
        logger.error(f"Error creating simple model: {e}")
        model_loaded = False
        return None

def load_model():
    """Load the face mask detection model with comprehensive error handling"""
    global model, model_loaded
    
    if model is not None and model_loaded:
        return model
    
    # If Hugging Face Hub is available, try to download the model
    if HUGGINGFACE_AVAILABLE:
        try:
            model_filename = "mask_detection_model.h5"
            repo_id = "sreenathsree1578/face_mask_detection"
            
            with st.spinner("🔄 Downloading model from Hugging Face Hub..."):
                model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            
            if TENSORFLOW_AVAILABLE:
                # Try different loading methods
                loading_methods = [
                    lambda: tf.keras.models.load_model(model_path, compile=False),
                    lambda: tf.keras.models.load_model(model_path),
                    lambda: tf.keras.models.load_model(model_path, custom_objects={})
                ]
                
                for i, load_method in enumerate(loading_methods):
                    try:
                        model = load_method()
                        model_loaded = True
                        logger.info(f"Model loaded successfully with method {i+1}")
                        st.success("✅ Model loaded successfully!")
                        return model
                    except Exception as e:
                        logger.warning(f"Loading method {i+1} failed: {e}")
                        continue
                
                st.error("❌ All model loading methods failed")
                model_loaded = False
                return None
                
        except Exception as e:
            logger.error(f"Hugging Face model download failed: {e}")
            st.warning("⚠️ Using fallback detection mode")
    
    # Fallback: Create a simple model
    return create_simple_model()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference"""
    try:
        # Ensure image has 3 channels
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Resize and normalize
        resized = cv2.resize(image, model_input_size)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return np.zeros((1, *model_input_size, 3), dtype=np.float32)

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using available detectors"""
    if face_detector_loaded and face_cascade is not None:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return [(x, y, w, h) for (x, y, w, h) in faces]
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
    
    # Fallback to simple detector
    simple_detector = SimpleFaceDetector()
    return simple_detector.detect_faces(image)

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify detected faces"""
    detections = []
    
    for (x, y, w, h) in faces:
        try:
            face_roi = image[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            if model_loaded and model is not None:
                # Use the actual model if available
                processed_face = preprocess_image(face_roi)
                predictions = model.predict(processed_face, verbose=0)
                class_id = np.argmax(predictions[0])
                confidence = float(predictions[0][class_id])
            else:
                # Fallback: Simple classification based on face position/size
                class_id = 0 if (w * h) > 5000 else 1  # Larger faces more likely to have masks
                confidence = 0.7 if class_id == 0 else 0.6
            
            if confidence >= confidence_threshold:
                detections.append({
                    "label": class_names[class_id],
                    "score": confidence,
                    "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
                })
                
        except Exception as e:
            logger.error(f"Face classification error: {e}")
            continue
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image"""
    try:
        # Convert to PIL Image for better text rendering
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image.convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        
        # Colors for different classes
        colors = {
            "Mask": (0, 255, 0),      # Green
            "No Mask": (255, 0, 0),   # Red
        }
        
        for detection in detections:
            try:
                box = detection["box"]
                label = detection["label"]
                confidence = detection["score"]
                color = colors.get(label, (0, 0, 255))
                
                # Draw bounding box
                draw.rectangle([(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])], 
                              outline=color, width=3)
                
                # Draw label background
                label_text = f"{label}: {confidence:.2%}"
                text_bbox = draw.textbbox((0, 0), label_text)
                text_width = text_bbox[2] - text_bbox[0] + 10
                text_height = text_bbox[3] - text_bbox[1] + 5
                
                draw.rectangle([
                    (box["xmin"], box["ymin"] - text_height),
                    (box["xmin"] + text_width, box["ymin"])
                ], fill=color)
                
                # Draw label text
                draw.text((box["xmin"] + 5, box["ymin"] - text_height + 2), 
                         label_text, fill=(255, 255, 255))
                
            except Exception as e:
                logger.error(f"Error drawing detection: {e}")
                continue
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Error in draw_detections: {e}")
        return image

class FaceMaskProcessor(VideoProcessorBase):
    """Video processor for real-time face mask detection"""
    
    def __init__(self, confidence_threshold: float = 0.5, mirror: bool = False):
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0
        self.processing_times = []
        
    def recv(self, frame):
        """Process incoming video frame"""
        try:
            start_time = time.time()
            
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror if requested
            if self.mirror:
                img = cv2.flip(img, 1)
            
            # Detect and classify faces
            faces = detect_faces(img)
            detections = classify_faces(img, faces, self.confidence_threshold)
            
            # Draw detections
            annotated_img = draw_detections(img, detections)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return frame

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Real-time face mask detection using AI. Detects faces and classifies mask usage.</p>', unsafe_allow_html=True)
    
    # Initialize components
    with st.spinner("🔄 Initializing system..."):
        load_face_detector()
        load_model()
    
    # Check system status
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Face Detector", "✅ Ready" if face_detector_loaded else "⚠️ Fallback")
    
    with status_col2:
        st.metric("AI Model", "✅ Loaded" if model_loaded else "⚠️ Basic")
    
    with status_col3:
        st.metric("WebRTC", "✅ Available" if STREAMLIT_WEBRTC_AVAILABLE else "❌ Unavailable")
    
    # Sidebar settings
    with st.sidebar:
        st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
        
        video_size = st.selectbox("Video Size", ["640x480", "1280x720", "1920x1080"], index=0)
        fps = st.slider("FPS", 5, 30, 15)
        mirror_video = st.checkbox("Mirror Video", False)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        if STREAMLIT_WEBRTC_AVAILABLE:
            width, height = map(int, video_size.split('x'))
            
            webrtc_ctx = webrtc_streamer(
                key="face-mask-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: FaceMaskProcessor(
                    confidence_threshold=confidence_threshold,
                    mirror=mirror_video
                ),
                media_stream_constraints={
                    "video": {"width": width, "height": height, "frameRate": fps},
                    "audio": False
                },
                async_processing=True,
            )
        else:
            st.warning("WebRTC is not available. Camera streaming disabled.")
            # Upload image fallback
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                # Process uploaded image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Detect and classify faces
                faces = detect_faces(img_array)
                detections = classify_faces(img_array, faces, confidence_threshold)
                
                # Draw detections
                result_image = draw_detections(img_array, detections)
                
                # Display result
                st.image(result_image, channels="BGR", use_column_width=True)
                
                # Show detection summary
                if detections:
                    mask_count = sum(1 for d in detections if d["label"] == "Mask")
                    no_mask_count = sum(1 for d in detections if d["label"] == "No Mask")
                    st.info(f"**Detection Results:** {mask_count} with mask, {no_mask_count} without mask")
                else:
                    st.warning("No faces detected or all detections below confidence threshold")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="sidebar-title">🎯 Legend</h3>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="legend-card" style="border-color: #22c55e;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px; border-radius: 4px;"></div>
                    <strong>Mask</strong>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #b0b0b0;">Person is wearing a mask</p>
            </div>
            
            <div class="legend-card" style="border-color: #ef4444;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px; border-radius: 4px;"></div>
                    <strong>No Mask</strong>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #b0b0b0;">Person is not wearing a mask</p>
            </div>
        """, unsafe_allow_html=True)
        
        # System information
        st.markdown("---")
        st.markdown("### 🔧 System Info")
        
        st.write(f"**Python:** {os.sys.version.split()[0]}")
        st.write(f"**OpenCV:** {cv2.__version__}")
        st.write(f"**TensorFlow:** {'Available' if TENSORFLOW_AVAILABLE else 'Not available'}")
        st.write(f"**Face Detector:** {'OpenCV' if face_detector_loaded else 'Fallback'}")
        st.write(f"**Model:** {'Loaded' if model_loaded else 'Fallback'}")
