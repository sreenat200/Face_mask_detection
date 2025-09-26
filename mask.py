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
import h5py
from huggingface_hub import hf_hub_download
import sys

# Set page config with transparent background
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix for OpenCV threading issues on Streamlit Cloud
cv2.setNumThreads(0)

# Set TensorFlow configurations for stability
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

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
        
        /* Streamlit widgets */
        .stSelectbox, .stSlider {
            color: white;
        }
        .css-1d391kg p {
            color: #b0b0b0;
        }
        .css-1d391kg label {
            color: #e0e0e0;
        }
        
        /* Footer */
        footer {
            color: #b0b0b0;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #333333;
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
        
        /* Error details */
        .error-details {
            background-color: #2a1a1a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #664444;
            font-family: monospace;
            font-size: 0.9rem;
            color: #ff9999;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False

@st.cache_resource
def load_model() -> Any:
    """Load the Keras face mask detection model from Hugging Face with enhanced error handling."""
    model_filename = "mask_detection_model.h5"
    repo_id = "sreenathsree1578/face_mask_detection"
    
    try:
        # Download model from Hugging Face Hub
        with st.spinner("Downloading model from Hugging Face Hub..."):
            model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        
        # Try loading with different methods
        try:
            # Method 1: Load without compilation to avoid issues
            model = tf.keras.models.load_model(model_path, compile=False)
            # Manually compile with simple settings
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy' if len(class_names) > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e1:
            st.error(f"Error loading model: {str(e1)}")
            return None
            
    except Exception as e:
        st.error(f"Error downloading model from Hugging Face: {str(e)}")
        return None

@st.cache_resource
def load_face_detector():
    """Load OpenCV's Haar cascade for face detection with error handling."""
    try:
        # Load the pre-trained Haar cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Check if cascade file exists
        if not os.path.exists(cascade_path):
            st.error(f"Haar cascade file not found at: {cascade_path}")
            return None
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if the cascade was loaded successfully
        if face_cascade.empty():
            st.error("Failed to load Haar cascade classifier")
            return None
        
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    try:
        # Ensure image is in the correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image, model_input_size, interpolation=cv2.INTER_AREA)
        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def detect_faces(image: np.ndarray, face_cascade) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade."""
    try:
        # Ensure image is valid
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces with more conservative parameters for stability
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples (x, y, w, h)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception as e:
        st.error(f"Error in face detection: {str(e)}")
        return []

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                  model, confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify each detected face as mask or no mask."""
    detections = []
    
    for (x, y, w, h) in faces:
        try:
            # Ensure coordinates are valid
            x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
            
            # Extract face ROI with bounds checking
            h_img, w_img = image.shape[:2]
            x_end = min(x + w, w_img)
            y_end = min(y + h, h_img)
            
            face_roi = image[y:y_end, x:x_end]
            
            # Skip if face ROI is empty or too small
            if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue
            
            # Preprocess the face ROI
            processed_face = preprocess_image(face_roi)
            if processed_face is None:
                continue
            
            # Classify the face
            predictions = model.predict(processed_face, verbose=0)
            
            # Handle different output formats
            if len(predictions.shape) == 2 and predictions.shape[1] == 2:
                # Binary classification with 2 outputs
                class_id = np.argmax(predictions[0])
                confidence = float(predictions[0][class_id])
            elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Binary classification with 1 output (sigmoid)
                confidence = float(predictions[0][0])
                class_id = 1 if confidence > 0.5 else 0
                confidence = confidence if class_id == 1 else (1.0 - confidence)
            else:
                # Fallback
                class_id = 0
                confidence = 0.5
            
            # Only add detection if confidence is above threshold
            if confidence >= confidence_threshold:
                detections.append({
                    "label": class_names[class_id],
                    "score": confidence,
                    "box": {"xmin": x, "ymin": y, "xmax": x_end, "ymax": y_end}
                })
        except Exception as e:
            st.warning(f"Error classifying face: {str(e)}")
            continue
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries
        
    Returns:
        Annotated image as numpy array
    """
    try:
        # Work directly with numpy array using OpenCV for better performance
        annotated_img = image.copy()
        
        # Define colors for different classes (BGR format for OpenCV)
        colors = {
            "Mask": (0, 255, 0),      # Green
            "No Mask": (0, 0, 255),   # Red
        }
        
        for detection in detections:
            try:
                # Get bounding box coordinates
                box = detection["box"]
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                
                # Get label and confidence
                label = detection["label"]
                confidence = detection["score"]
                
                # Get color based on label
                color = colors.get(label, (255, 0, 0))  # Default to blue if label not found
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Create label text with confidence
                label_text = f"{label}: {confidence:.2%}"
                
                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw filled rectangle for text background
                cv2.rectangle(annotated_img, 
                            (xmin, ymin - text_height - baseline - 5), 
                            (xmin + text_width, ymin), 
                            color, -1)
                
                # Draw text
                cv2.putText(annotated_img, label_text, (xmin, ymin - baseline - 5), 
                          font, font_scale, (255, 255, 255), thickness)
            except Exception as e:
                continue
        
        return annotated_img
    except Exception as e:
        st.error(f"Error drawing detections: {str(e)}")
        return image

class FaceMaskProcessor(VideoProcessorBase):
    """Video processor class for real-time face mask detection."""
    
    def __init__(self, model: Any, face_cascade, target_size: Tuple[int, int] = (640, 480), 
                 confidence_threshold: float = 0.5, mirror: bool = False):
        self.model = model
        self.face_cascade = face_cascade
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0
        self.processing_times = []
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame."""
        try:
            start_time = time.time()
            
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror the image if requested
            if self.mirror:
                img = cv2.flip(img, 1)
            
            # Resize frame if needed
            if img.shape[:2][::-1] != self.target_size:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Only process every few frames to reduce load
            if self.frame_count % 3 == 0:  # Process every 3rd frame
                # Detect faces
                faces = detect_faces(img, self.face_cascade)
                
                # Classify each detected face
                detections = classify_faces(img, faces, self.model, self.confidence_threshold)
                
                # Draw detections on frame
                annotated_img = draw_detections(img, detections)
            else:
                annotated_img = img
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:  # Keep last 30 measurements
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            # Return original frame on error
            return frame
    
    def get_average_fps(self) -> float:
        """Calculate average FPS based on processing times."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def main():
    """Main function to run the Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Real-time face mask detection using a Keras model. The system detects faces and classifies whether they are wearing a mask or not.</p>', unsafe_allow_html=True)
    
    # Load model and face detector
    with st.spinner("Loading AI models..."):
        model = load_model()
        face_cascade = load_face_detector()
    
    # Check if models loaded successfully
    if model is None or face_cascade is None:
        st.error("Failed to load the model or face detector. Please check your internet connection and try refreshing the page.")
        
        # Additional debugging information
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">🔍 Debugging Information</h3>', unsafe_allow_html=True)
        
        st.write("**System Information:**")
        st.write(f"- Python Version: {sys.version}")
        st.write(f"- TensorFlow Version: {tf.__version__}")
        st.write(f"- OpenCV Version: {cv2.__version__}")
        st.write(f"- Model Status: {'✅ Loaded' if model is not None else '❌ Failed'}")
        st.write(f"- Face Detector Status: {'✅ Loaded' if face_cascade is not None else '❌ Failed'}")
        
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
        
        # Video size selection
        video_size = st.selectbox(
            "Video Size",
            options=["640x480", "800x600"],  # Reduced options for stability
            index=0,
            help="Select the resolution for the video stream"
        )
        
        # Mirror video option
        mirror_video = st.checkbox(
            "Mirror Video",
            value=True,
            help="Flip the video horizontally"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Minimum confidence score for detections"
        )
    
    # Parse video size
    width, height = map(int, video_size.split('x'))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="face-mask-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: FaceMaskProcessor(
                model, face_cascade, (width, height), confidence_threshold, mirror_video
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": width},
                    "height": {"ideal": height},
                    "frameRate": {"ideal": 15}  # Conservative frame rate
                },
                "audio": False
            },
            async_processing=True,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Instructions
        st.info("""
            **Instructions:**
            1. Click "START" to begin video streaming
            2. Allow camera access when prompted
            3. The system will detect faces and classify mask usage in real-time
            4. Green boxes = With mask, Red boxes = Without mask
        """)
    
    with col2:
        st.markdown('<h3 class="sidebar-title">🎯 Detection Legend</h3>', unsafe_allow_html=True)
        
        # Create legend cards
        st.markdown("""
            <div class="legend-card" style="border-color: #22c55e;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px; border-radius: 4px;"></div>
                    <strong>Mask</strong>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is wearing a mask</p>
            </div>
            
            <div class="legend-card" style="border-color: #ef4444;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px; border-radius: 4px;"></div>
                    <strong>No Mask</strong>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is not wearing a mask</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Performance tips
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">💡 Tips</h3>', unsafe_allow_html=True)
        st.markdown("""
        - Ensure good lighting for better detection
        - Face the camera directly
        - Stay within 1-3 feet of the camera
        - If the app becomes slow, try refreshing the page
        """)
    
    # System information at the bottom
    st.markdown("---")
    st.markdown("""
        <div class="system-info">
            <h3>System Information</h3>
            <p>TensorFlow Version: {tf_version}</p>
            <p>OpenCV Version: {cv_version}</p>
            <p>Model: Loaded from Hugging Face Hub</p>
            <p>Face Detector: Haar Cascade (CPU optimized)</p>
            <p>Device: CPU Only</p>
        </div>
    """.format(
        tf_version=tf.__version__,
        cv_version=cv2.__version__
    ), unsafe_allow_html=True)
    
    # Footer
    st.markdown(
        '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">'
        'Built with ❤️ using Streamlit, TensorFlow, and OpenCV'
        '</footer>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
