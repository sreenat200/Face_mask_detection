import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from typing import Tuple, List, Dict, Any
import os
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    CV2_AVAILABLE = False

try:
    import tensorflow as tf
    # Force CPU usage to avoid GPU issues
    tf.config.set_visible_devices([], 'GPU')
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Custom CSS for modern styling
st.markdown("""
    <style>
        .main { background-color: #121212; }
        .css-1d391kg { background-color: #1e1e1e; }
        .main-header { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; }
        .description { font-size: 1.1rem; color: #b0b0b0; margin-bottom: 2rem; }
        .sidebar-title { font-size: 1.3rem; font-weight: 600; color: #ffffff; margin-bottom: 1rem; }
        .video-container { border-radius: 1rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); overflow: hidden; background-color: #1e1e1e; padding: 1rem; border: 1px solid #333333; }
        .stButton button { background-color: #4a4a4a; color: white; border: none; border-radius: 0.5rem; padding: 0.5rem 1rem; font-weight: 600; }
        .stButton button:hover { background-color: #5a5a5a; }
        .legend-card { background-color: #2a2a2a; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid; }
        .system-info { background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; border: 1px solid #333333; font-size: 0.8rem; color: #888; }
        .error-details { background-color: #2a1a1a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #664444; font-family: monospace; color: #ff9999; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

# Global variables
model = None
face_cascade = None
model_input_size = (128, 128)
class_names = ['Mask', 'No Mask']
model_loaded = False
face_detector_loaded = False

def load_face_detector():
    """Load OpenCV's Haar cascade for face detection."""
    global face_cascade, face_detector_loaded
    
    if not CV2_AVAILABLE:
        st.error("OpenCV is not available. Cannot load face detector.")
        return False
        
    try:
        # Try multiple cascade paths
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            './haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            try:
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    face_detector_loaded = True
                    st.success(f"Face detector loaded from: {path}")
                    return True
            except Exception as e:
                continue
                
        # If no path worked, try to download the cascade
        st.warning("Trying to download face detection model...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            if not face_cascade.empty():
                face_detector_loaded = True
                st.success("Face detector downloaded and loaded successfully!")
                return True
        except Exception as e:
            st.error(f"Failed to download face detector: {e}")
            
        face_detector_loaded = False
        return False
        
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        face_detector_loaded = False
        return False

def load_model():
    """Load the Keras face mask detection model with multiple fallback strategies."""
    global model, model_loaded
    
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow is not available. Cannot load model.")
        return None
        
    if model is not None and model_loaded:
        return model
        
    model_filename = "mask_detection_model.h5"
    repo_id = "sreenathsree1578/face_mask_detection"
    
    try:
        with st.spinner("🔄 Downloading model from Hugging Face Hub..."):
            model_path = hf_hub_download(
                repo_id=repo_id, 
                filename=model_filename,
                cache_dir="./models"
            )
        st.success("✅ Model downloaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return load_alternative_model()
    
    # Try different loading methods
    loading_methods = [
        {"name": "Standard load", "kwargs": {}},
        {"name": "Load without compilation", "kwargs": {"compile": False}},
        {"name": "Load with custom InputLayer", "kwargs": {"compile": False, "custom_objects": {"InputLayer": tf.keras.layers.InputLayer}}},
    ]
    
    for method in loading_methods:
        try:
            st.info(f"Trying {method['name']}...")
            model = tf.keras.models.load_model(model_path, **method['kwargs'])
            model_loaded = True
            st.success(f"✅ Model loaded successfully using {method['name']}!")
            
            # Test the model with a dummy prediction
            try:
                dummy_input = np.random.random((1, *model_input_size, 3)).astype(np.float32)
                prediction = model.predict(dummy_input, verbose=0)
                st.success("✅ Model test prediction successful!")
            except Exception as e:
                st.warning(f"⚠️ Model test prediction failed: {e}")
                
            return model
            
        except Exception as e:
            st.error(f"❌ {method['name']} failed: {str(e)}")
            continue
    
    st.error("All loading methods failed. Trying alternative approach...")
    return load_alternative_model()

def load_alternative_model():
    """Create a fallback model if the main model fails to load."""
    global model, model_loaded
    
    try:
        st.info("🔄 Creating a simple fallback model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_loaded = True
        st.warning("⚠️ Using fallback model (limited functionality)")
        return model
        
    except Exception as e:
        st.error(f"❌ Fallback model creation failed: {e}")
        model_loaded = False
        return None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    try:
        if not CV2_AVAILABLE:
            return np.zeros((1, *model_input_size, 3), dtype=np.float32)
            
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image, model_input_size)
        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return np.zeros((1, *model_input_size, 3), dtype=np.float32)

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade."""
    if not face_detector_loaded or not CV2_AVAILABLE:
        h, w = image.shape[:2]
        return [(w//4, h//4, w//2, h//2)]
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    except Exception as e:
        st.warning(f"Face detection error: {e}")
        return []

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify each detected face as mask or no mask."""
    if not model_loaded:
        detections = []
        for i, (x, y, w, h) in enumerate(faces):
            detections.append({
                "label": "Mask" if i % 2 == 0 else "No Mask",
                "score": 0.85 if i % 2 == 0 else 0.75,
                "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
            })
        return detections
    
    detections = []
    
    for (x, y, w, h) in faces:
        try:
            face_roi = image[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            processed_face = preprocess_image(face_roi)
            predictions = model.predict(processed_face, verbose=0)
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id])
            
            if confidence >= confidence_threshold:
                detections.append({
                    "label": class_names[class_id],
                    "score": confidence,
                    "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
                })
        except Exception as e:
            st.warning(f"Error classifying face: {str(e)}")
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
            
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = {
            "Mask": (0, 255, 0),      # Green
            "No Mask": (255, 0, 0),   # Red
        }
        
        for detection in detections:
            box = detection["box"]
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            label = detection["label"]
            confidence = detection["score"]
            color = colors.get(label, (0, 0, 255))
            
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
            label_text = f"{label}: {confidence:.2%}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle(
                [(xmin, ymin - text_height - 5), (xmin + text_width + 10, ymin - 5)],
                fill=color
            )
            draw.text((xmin + 5, ymin - text_height - 5), label_text, fill="white", font=font)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error drawing detections: {e}")
        return image

class FaceMaskProcessor(VideoProcessorBase):
    """Video processor class for real-time face mask detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480), 
                 confidence_threshold: float = 0.5, mirror: bool = False):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0
        self.processing_times = []
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame."""
        if not AV_AVAILABLE:
            return frame
            
        start_time = time.time()
        
        try:
            img = frame.to_ndarray(format="bgr24")
            if self.mirror:
                img = cv2.flip(img, 1)
            if img.shape[:2][::-1] != self.target_size:
                img = cv2.resize(img, self.target_size)
            
            faces = detect_faces(img)
            detections = classify_faces(img, faces, self.confidence_threshold)
            annotated_img = draw_detections(img, detections)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def get_average_fps(self) -> float:
        """Calculate average FPS based on processing times."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Real-time face mask detection using deep learning. Detects faces and classifies mask usage.</p>', unsafe_allow_html=True)
    
    # Dependency check
    st.sidebar.markdown('<h3 class="sidebar-title">🔧 System Status</h3>', unsafe_allow_html=True)
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        st.write("**Dependencies:**")
        st.write(f"OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
        st.write(f"TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
        st.write(f"WebRTC: {'✅' if WEBRTC_AVAILABLE else '❌'}")
    
    with status_col2:
        st.write("**Components:**")
        st.write(f"Face Detector: {'✅' if face_detector_loaded else '❌'}")
        st.write(f"ML Model: {'✅' if model_loaded else '❌'}")
    
    # Load models
    with st.spinner("Loading models and detectors..."):
        load_face_detector()
        load_model()
    
    if not CV2_AVAILABLE or not TENSORFLOW_AVAILABLE:
        st.error("""
        **Critical dependencies missing!**
        
        Required packages:
        - OpenCV (for image processing)
        - TensorFlow (for ML model)
        
        Install with: `pip install opencv-python-headless tensorflow==2.10.0`
        """)
        return
    
    if not face_detector_loaded or not model_loaded:
        st.warning("""
        **Some components failed to load, but the app will run in demonstration mode.**
        - You'll see sample detections
        - Full functionality requires successful model loading
        """)
    
    # Settings sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
    
    video_size = st.sidebar.selectbox(
        "Video Size",
        options=["640x480", "1280x720", "800x600"],
        index=0
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    mirror_video = st.sidebar.checkbox("Mirror Video", value=True)
    
    width, height = map(int, video_size.split('x'))
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        if WEBRTC_AVAILABLE and AV_AVAILABLE:
            webrtc_ctx = webrtc_streamer(
                key="face-mask-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: FaceMaskProcessor(
                    target_size=(width, height),
                    confidence_threshold=confidence_threshold,
                    mirror=mirror_video
                ),
                media_stream_constraints={
                    "video": {"width": width, "height": height, "frameRate": 15},
                    "audio": False
                },
                async_processing=True,
            )
        else:
            st.warning("WebRTC not available. Camera streaming disabled.")
            sample_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            faces = detect_faces(sample_img)
            detections = classify_faces(sample_img, faces, confidence_threshold)
            result_img = draw_detections(sample_img, detections)
            st.image(result_img, channels="BGR", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("""
        **Instructions:**
        1. Click **START** to begin camera streaming
        2. Allow camera access when prompted  
        3. System will detect faces and classify mask usage
        4. **Green** = With mask, **Red** = Without mask
        """)
    
    with col2:
        st.markdown('<h3 class="sidebar-title">🎯 Detection Legend</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="legend-card" style="border-color: #22c55e;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px; border-radius: 4px;"></div>
                <strong>Mask Detected</strong>
            </div>
            <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is wearing a mask properly</p>
        </div>
        
        <div class="legend-card" style="border-color: #ef4444;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px; border-radius: 4px;"></div>
                <strong>No Mask</strong>
            </div>
            <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is not wearing a mask</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">📊 Statistics</h3>', unsafe_allow_html=True)
        st.write(f"**Model Input Size:** {model_input_size[0]}x{model_input_size[1]}")
        st.write(f"**Classes:** {', '.join(class_names)}")
        st.write(f"**Confidence Threshold:** {confidence_threshold:.0%}")
    
    st.markdown("---")
    st.markdown("""
    <div class="system-info">
        <h3>System Information</h3>
        <p><strong>Model:</strong> Face Mask Detection CNN</p>
        <p><strong>Face Detection:</strong> Haar Cascade Classifier</p>
        <p><strong>Framework:</strong> TensorFlow + OpenCV</p>
        <p><strong>Status:</strong> {'Fully Operational' if all([CV2_AVAILABLE, TENSORFLOW_AVAILABLE, face_detector_loaded, model_loaded]) else 'Limited Functionality'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 Debug Information"):
        st.write("**Environment Info:**")
        st.write(f"- Python: {os.sys.version.split()[0]}")
        if TENSORFLOW_AVAILABLE:
            st.write(f"- TensorFlow: {tf.__version__}")
        if CV2_AVAILABLE:
            st.write(f"- OpenCV: {cv2.__version__}")
        
        st.write("**File Structure:**")
        for file in os.listdir('.'):
            if file.endswith(('.py', '.txt', '.h5', '.xml')):
                st.write(f"- {file}")

if __name__ == "__main__":
    main()
