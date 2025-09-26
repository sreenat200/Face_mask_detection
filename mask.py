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
import requests
import tempfile

# Set page config with transparent background
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

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
error_details = []

def add_error(message):
    """Add error message to the error details list."""
    error_details.append(f"[{time.strftime('%H:%M:%S')}] {message}")

def inspect_model_file(model_path):
    """Inspect the model file structure to understand its format."""
    try:
        with h5py.File(model_path, 'r') as f:
            # Check if it's a valid HDF5 file
            st.write("Model file structure:")
            def print_attrs(name, obj):
                st.write(f"{name}: {list(obj.attrs.keys())}")
            f.visititems(print_attrs)
            
            # Check for model weights
            if 'model_weights' in f:
                st.write("Found model_weights group")
            if 'model_config' in f:
                st.write("Found model_config")
            if 'training_config' in f:
                st.write("Found training_config")
                
    except Exception as e:
        st.error(f"Error inspecting model file: {str(e)}")
        add_error(f"Model inspection error: {str(e)}")

def load_model() -> Any:
    """Load the Keras face mask detection model with enhanced error handling."""
    global model, model_loaded
    if model is None:
        model_filename = "mask_detection_model.h5"
        repo_id = "sreenathsree1578/face_mask_detection"
        
        try:
            # Check network connection
            add_error("Checking network connection...")
            try:
                response = requests.get("https://huggingface.co", timeout=5)
                if response.status_code != 200:
                    add_error(f"Network error: Hugging Face returned status {response.status_code}")
                    st.error("Cannot connect to Hugging Face Hub")
                    return None
            except Exception as e:
                add_error(f"Network connection error: {str(e)}")
                st.error(f"Network connection error: {str(e)}")
                return None
                
            # Download model
            add_error("Downloading model from Hugging Face Hub...")
            with st.spinner("Downloading model from Hugging Face Hub..."):
                try:
                    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
                    add_error(f"Model downloaded to: {model_path}")
                    st.write(f"Model downloaded to: {model_path}")
                except Exception as e:
                    add_error(f"Download failed: {str(e)}")
                    st.error(f"Failed to download model: {str(e)}")
                    return None
            
            # Check file exists
            if not os.path.exists(model_path):
                add_error(f"Model file not found: {model_path}")
                st.error(f"Model file not found: {model_path}")
                return None
                
            # Check file size
            file_size = os.path.getsize(model_path)
            add_error(f"Model file size: {file_size / (1024*1024):.2f} MB")
            st.write(f"Model file size: {file_size / (1024*1024):.2f} MB")
            
            # Try loading with different methods
            add_error("Attempting to load model...")
            st.write("Attempting to load model...")
            
            # Method 1: Standard Keras load
            try:
                model = tf.keras.models.load_model(model_path)
                model_loaded = True
                add_error("Model loaded successfully with standard method")
                st.success("Model loaded successfully!")
                return model
            except Exception as e1:
                add_error(f"Standard load failed: {str(e1)}")
                st.warning(f"Standard load failed: {str(e1)}")
                
                # Method 2: Try with custom objects
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model_loaded = True
                    add_error("Model loaded successfully with compile=False")
                    st.success("Model loaded successfully (without compilation)!")
                    return model
                except Exception as e2:
                    add_error(f"Compile=False load failed: {str(e2)}")
                    st.warning(f"Compile=False load failed: {str(e2)}")
                    
                    # Method 3: Try loading as SavedModel if it's actually a directory
                    try:
                        if os.path.isdir(model_path):
                            model = tf.keras.models.load_model(model_path)
                            model_loaded = True
                            add_error("Model loaded successfully as SavedModel")
                            st.success("Model loaded successfully as SavedModel!")
                            return model
                    except Exception as e3:
                        add_error(f"SavedModel load failed: {str(e3)}")
                        st.warning(f"SavedModel load failed: {str(e3)}")
            
            # If all methods failed
            add_error("All model loading methods failed")
            st.error("All model loading methods failed")
            model_loaded = False
            return None
            
        except Exception as e:
            add_error(f"Unexpected error loading model: {str(e)}")
            st.error(f"Unexpected error loading model: {str(e)}")
            model_loaded = False
            return None
    return model

def load_face_detector():
    """Load OpenCV's Haar cascade for face detection."""
    global face_cascade, face_detector_loaded
    if face_cascade is None:
        try:
            add_error("Loading face detector...")
            st.write("Loading face detector...")
            
            # Check OpenCV version
            st.write(f"OpenCV version: {cv2.__version__}")
            
            # Try to load Haar cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            add_error(f"Cascade path: {cascade_path}")
            st.write(f"Cascade path: {cascade_path}")
            
            if not os.path.exists(cascade_path):
                add_error(f"Cascade file not found: {cascade_path}")
                st.error(f"Cascade file not found: {cascade_path}")
                face_detector_loaded = False
                return False
                
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                add_error("Cascade classifier is empty")
                st.error("Cascade classifier is empty")
                face_detector_loaded = False
                return False
            
            face_detector_loaded = True
            add_error("Face detector loaded successfully")
            st.success("Face detector loaded successfully!")
            return True
        except Exception as e:
            add_error(f"Error loading face detector: {str(e)}")
            st.error(f"Error loading face detector: {str(e)}")
            face_detector_loaded = False
            return False
    return True

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    # Resize to model input size
    resized = cv2.resize(image, model_input_size)
    # Normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade."""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert to list of tuples (x, y, w, h)
    return [(x, y, w, h) for (x, y, w, h) in faces]

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify each detected face as mask or no mask."""
    detections = []
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Skip if face ROI is empty
        if face_roi.size == 0:
            continue
        
        # Preprocess the face ROI
        processed_face = preprocess_image(face_roi)
        
        # Classify the face
        try:
            predictions = model.predict(processed_face, verbose=0)
            
            # Get the class with the highest probability
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id])
            
            # Only add detection if confidence is above threshold
            if confidence >= confidence_threshold:
                detections.append({
                    "label": class_names[class_id],
                    "score": confidence,
                    "box": {"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
                })
        except Exception as e:
            st.warning(f"Error classifying face: {str(e)}")
            add_error(f"Face classification error: {str(e)}")
    
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
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Define colors for different classes
    colors = {
        "Mask": (0, 255, 0),      # Green
        "No Mask": (255, 0, 0),   # Red
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
            color = colors.get(label, (0, 0, 255))  # Default to blue if label not found
            
            # Draw bounding box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
            
            # Create label text with confidence
            label_text = f"{label}: {confidence:.2%}"
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw filled rectangle for text background
            draw.rectangle(
                [(xmin, ymin - text_height - 5), (xmin + text_width + 10, ymin - 5)],
                fill=color
            )
            
            # Draw text
            draw.text((xmin + 5, ymin - text_height - 5), label_text, fill="white", font=font)
        except Exception as e:
            st.warning(f"Error drawing detection: {str(e)}")
            add_error(f"Drawing detection error: {str(e)}")
    
    # Convert back to numpy array
    return np.array(pil_image)

class FaceMaskProcessor(VideoProcessorBase):
    """Video processor class for real-time face mask detection."""
    
    def __init__(self, model: Any, target_size: Tuple[int, int] = (640, 480), 
                 confidence_threshold: float = 0.5, mirror: bool = False):
        self.model = model
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0
        self.processing_times = []
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame."""
        start_time = time.time()
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror the image if requested
        if self.mirror:
            img = cv2.flip(img, 1)
        
        # Resize frame if needed
        if img.shape[:2][::-1] != self.target_size:
            img = cv2.resize(img, self.target_size)
        
        # Detect faces
        faces = detect_faces(img)
        
        # Classify each detected face
        detections = classify_faces(img, faces, self.confidence_threshold)
        
        # Draw detections on frame
        annotated_img = draw_detections(img, detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:  # Keep last 30 measurements
            self.processing_times.pop(0)
        
        self.frame_count += 1
        
        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
    
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
    model = load_model()
    load_face_detector()
    
    # Check if models loaded successfully
    if not model_loaded or not face_detector_loaded:
        st.error("Failed to load the model or face detector. Please check the files and try again.")
        
        # Additional debugging information
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">🔍 Debugging Information</h3>', unsafe_allow_html=True)
        
        st.write("**Current Directory:**", os.getcwd())
        st.write("**Files in Directory:**")
        for file in os.listdir():
            if file.endswith(('.h5', '.keras')):
                st.write(f"- {file}")
        
        # Show system information
        st.write("**System Information:**")
        st.write(f"- Python Version: {os.sys.version}")
        st.write(f"- TensorFlow Version: {tf.__version__}")
        st.write(f"- OpenCV Version: {cv2.__version__}")
        st.write(f"- Streamlit Version: {st.__version__}")
        st.write(f"- Hugging Face Hub Version: {hf_hub_download.__module__}")
        
        # Show error details
        if error_details:
            st.markdown('<h3 class="sidebar-title">❌ Error Details</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="error-details">{chr(10).join(error_details)}</div>', unsafe_allow_html=True)
        
        # Check model file integrity
        model_path = "mask_detection_model.h5"
        if os.path.exists(model_path):
            st.write(f"\n**Model File Information:**")
            st.write(f"- File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            st.write(f"- File exists: Yes")
            
            # Try to read the file as HDF5
            try:
                with h5py.File(model_path, 'r') as f:
                    st.write(f"- HDF5 file: Valid")
                    st.write(f"- Root keys: {list(f.keys())}")
            except Exception as e:
                st.write(f"- HDF5 file: Invalid - {str(e)}")
        
        # Provide alternative solutions
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">🛠️ Alternative Solutions</h3>', unsafe_allow_html=True)
        
        st.info("""
            **If the model loading continues to fail:**
            1. Check your internet connection
            2. Try downloading the model manually from Hugging Face
            3. Place the model file in the same directory as this script
            4. Verify the model file is not corrupted
        """)
        
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
        
        # Video size selection
        video_size = st.selectbox(
            "Video Size",
            options=["640x480", "1280x720", "1920x1080"],
            index=0,
            help="Select the resolution for the video stream"
        )
        
        # FPS selection
        fps = st.slider(
            "Frames Per Second (FPS)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help="Adjust the frame rate for video processing"
        )
        
        # Mirror video option
        mirror_video = st.checkbox(
            "Mirror Video",
            value=False,
            help="Flip the video horizontally"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Face detection parameters
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">🔍 Face Detection</h3>', unsafe_allow_html=True)
        
        scale_factor = st.slider(
            "Scale Factor",
            min_value=1.01,
            max_value=1.5,
            value=1.1,
            step=0.01,
            help="Parameter specifying how much the image size is reduced at each image scale"
        )
        
        min_neighbors = st.slider(
            "Min Neighbors",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Parameter specifying how many neighbors each candidate rectangle should have to retain it"
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
                model, (width, height), confidence_threshold, mirror_video
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": width},
                    "height": {"ideal": height},
                    "frameRate": {"ideal": fps}
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
    
    # System information at the bottom
    st.markdown("---")
    st.markdown("""
        <div class="system-info">
            <h3>System Information</h3>
            <p>TensorFlow Version: {tf_version}</p>
            <p>Model: {model_name} (Loaded from Hugging Face)</p>
            <p>Face Detector: {detector_status}</p>
            <p>Device: CPU</p>
        </div>
    """.format(
        tf_version=tf.__version__,
        model_name="mask_detection_model.h5",
        detector_status="Loaded" if face_detector_loaded else "Failed to load"
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
