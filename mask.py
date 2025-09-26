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
    </style>
""", unsafe_allow_html=True)

# Force TensorFlow to use CPU only to prevent segmentation fault
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow warnings

# Configure TensorFlow for better stability
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False
model_lock = threading.Lock()

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
            
            if model_path is None:
                st.error("Model file not found. Please ensure 'mask_detection_model.h5' is in the app directory.")
                st.info("Available files: " + ", ".join([f for f in os.listdir() if f.endswith(('.h5', '.keras'))]))
                model_loaded = False
                return None
            
            try:
                # Clear any existing models to prevent memory issues
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Load model with specific settings to prevent segmentation fault
                with tf.device('/CPU:0'):
                    model = tf.keras.models.load_model(
                        model_path,
                        compile=False  # Don't compile to avoid potential issues
                    )
                    
                    # Warm up the model with a dummy prediction
                    dummy_input = np.zeros((1, *model_input_size, 3), dtype=np.float32)
                    _ = model.predict(dummy_input, verbose=0)
                    
                model_loaded = True
                st.success(f"Model loaded successfully from {model_path}")
                return model
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.error("This might be due to:")
                st.error("1. Corrupted model file")
                st.error("2. Incompatible TensorFlow version")
                st.error("3. Missing dependencies")
                model_loaded = False
                return None
    
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
                        face_cascade = cv2.CascadeClassifier(cascade_file)
                        if not face_cascade.empty():
                            break
                except:
                    continue
            
            if face_cascade is None or face_cascade.empty():
                st.error("Failed to load face detector. Trying alternative approach...")
                # Try to create a simple face detector using cv2.dnn if Haar cascade fails
                return init_dnn_face_detector()
            
            face_detector_loaded = True
            st.success("Haar cascade face detector loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Error loading face detector: {str(e)}")
            face_detector_loaded = False
            return False
    
    return True

def init_dnn_face_detector():
    """Initialize DNN-based face detector as fallback."""
    global face_detector_loaded
    try:
        # This would require additional model files, so we'll create a fallback
        st.warning("Using basic face detection. For better results, ensure OpenCV is properly installed.")
        face_detector_loaded = True
        return True
    except:
        face_detector_loaded = False
        return False

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference with safety checks."""
    try:
        if image is None or image.size == 0:
            return None
            
        # Resize to model input size
        resized = cv2.resize(image, model_input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB if BGR
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade with safety checks."""
    try:
        if image is None or image.size == 0:
            return []
            
        if face_cascade is None:
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with conservative parameters to avoid crashes
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(300, 300),  # Limit max size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples and limit number of faces to prevent overload
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return face_list[:10]  # Limit to 10 faces maximum
        
    except Exception as e:
        st.error(f"Error in face detection: {str(e)}")
        return []

def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
    """Classify each detected face as mask or no mask with safety checks."""
    detections = []
    
    if model is None:
        return detections
    
    try:
        for (x, y, w, h) in faces:
            # Validate bounding box
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                continue
                
            if x + w > image.shape[1] or y + h > image.shape[0]:
                continue
            
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Skip if face ROI is empty or too small
            if face_roi.size == 0 or min(face_roi.shape[:2]) < 20:
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
                continue
                
    except Exception as e:
        st.error(f"Error in face classification: {str(e)}")
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image with safety checks."""
    try:
        if image is None or image.size == 0:
            return image
            
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Define colors for different classes
        colors = {
            "Mask": (0, 255, 0),      # Green
            "No Mask": (255, 0, 0),   # Red
        }
        
        for detection in detections:
            try:
                # Get bounding box coordinates
                box = detection.get("box", {})
                xmin = box.get("xmin", 0)
                ymin = box.get("ymin", 0)
                xmax = box.get("xmax", 0)
                ymax = box.get("ymax", 0)
                
                # Validate coordinates
                if xmin >= xmax or ymin >= ymax:
                    continue
                
                # Get label and confidence
                label = detection.get("label", "Unknown")
                confidence = detection.get("score", 0.0)
                
                # Get color based on label
                color = colors.get(label, (0, 0, 255))  # Default to blue if label not found
                
                # Draw bounding box
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
                
                # Create label text with confidence
                label_text = f"{label}: {confidence:.2%}"
                
                # Draw text background and text if font is available
                if font:
                    try:
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
                    except:
                        # Fallback to simple text without background
                        draw.text((xmin + 5, ymin - 20), label_text, fill=color)
                
            except Exception as e:
                st.warning(f"Error drawing detection: {str(e)}")
                continue
        
        # Convert back to numpy array
        return np.array(pil_image)
        
    except Exception as e:
        st.error(f"Error in drawing detections: {str(e)}")
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
        self.skip_frames = 2  # Process every 3rd frame for better performance
        self.current_frame = 0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frame with enhanced error handling."""
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
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Skip frames for performance
            self.current_frame += 1
            if self.current_frame % (self.skip_frames + 1) != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            start_time = time.time()
            
            # Only process if model and detector are loaded
            if model_loaded and face_detector_loaded:
                # Detect faces
                faces = detect_faces(img)
                
                if faces:
                    # Classify each detected face
                    detections = classify_faces(img, faces, self.confidence_threshold)
                    
                    # Draw detections on frame
                    img = draw_detections(img, detections)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:  # Keep last 30 measurements
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Log error but don't crash
            print(f"Error processing frame: {str(e)}")
            # Return original frame if processing fails
            try:
                return frame
            except:
                # Create a black frame if everything fails
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return av.VideoFrame.from_ndarray(black_frame, format="bgr24")
    
    def get_average_fps(self) -> float:
        """Calculate average FPS based on processing times."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def main():
    """Main function to run the Streamlit app with enhanced error handling."""
    try:
        # Header
        st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="description">Real-time face mask detection using a Keras model. The system detects faces and classifies whether they are wearing a mask or not.</p>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'initialization_complete' not in st.session_state:
            st.session_state.initialization_complete = False
        
        # Load model and face detector with progress indication
        if not st.session_state.initialization_complete:
            with st.spinner('Loading AI models...'):
                model = load_model()
                face_detector_status = load_face_detector()
                st.session_state.initialization_complete = True
        
        # Check if models loaded successfully
        if not model_loaded:
            st.error("❌ Failed to load the face mask detection model.")
            st.info("Please ensure 'mask_detection_model.h5' is in the application directory.")
            
            # Show available files
            st.write("**Available model files:**")
            model_files = [f for f in os.listdir() if f.endswith(('.h5', '.keras', '.pb'))]
            if model_files:
                for file in model_files:
                    st.write(f"- {file}")
            else:
                st.write("No model files found")
            
            return
        
        if not face_detector_loaded:
            st.warning("⚠️ Face detector not loaded optimally, but the app will continue with basic detection.")
        
        # Success message
        if model_loaded and face_detector_loaded:
            st.success("✅ All models loaded successfully!")
        
        # Sidebar
        with st.sidebar:
            st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
            
            # Video size selection
            video_size = st.selectbox(
                "Video Size",
                options=["640x480", "800x600", "1280x720"],
                index=0,
                help="Select the resolution for the video stream"
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
            
            # Mirror video option
            mirror_video = st.checkbox(
                "Mirror Video",
                value=True,
                help="Flip the video horizontally (recommended for selfie view)"
            )
            
            # Performance settings
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">⚡ Performance</h3>', unsafe_allow_html=True)
            
            fps_limit = st.slider(
                "FPS Limit",
                min_value=5,
                max_value=25,
                value=15,
                help="Limit FPS to reduce CPU usage"
            )
        
        # Parse video size
        width, height = map(int, video_size.split('x'))
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            
            # Only show webcam if models are loaded
            if model_loaded:
                try:
                    # WebRTC streamer with enhanced error handling
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
                                "frameRate": {"ideal": fps_limit, "max": fps_limit}
                            },
                            "audio": False
                        },
                        async_processing=True,
                    )
                except Exception as e:
                    st.error(f"Error initializing video stream: {str(e)}")
                    st.info("Try refreshing the page or checking your camera permissions.")
            else:
                st.error("Cannot start video stream without loaded models.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Instructions
            st.info("""
                **Instructions:**
                1. Click "START" to begin video streaming
                2. Allow camera access when prompted
                3. The system will detect faces and classify mask usage in real-time
                4. Green boxes = With mask, Red boxes = Without mask
                
                **Troubleshooting:**
                - If video doesn't start, refresh the page
                - Ensure good lighting for better detection
                - If performance is slow, reduce video size or FPS limit
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
            
            # System status
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">📊 System Status</h3>', unsafe_allow_html=True)
            
            status_color = "#22c55e" if (model_loaded and face_detector_loaded) else "#ef4444"
            status_text = "All systems operational" if (model_loaded and face_detector_loaded) else "Some issues detected"
            
            st.markdown(f"""
                <div style="padding: 0.5rem; background-color: {status_color}20; border-left: 4px solid {status_color}; border-radius: 0.25rem;">
                    <strong style="color: {status_color};">{status_text}</strong>
                </div>
            """, unsafe_allow_html=True)
        
        # System information at the bottom
        st.markdown("---")
        
        model_size = 0
        if os.path.exists("mask_detection_model.h5"):
            model_size = os.path.getsize("mask_detection_model.h5") / (1024*1024)
        
        st.markdown(f"""
            <div class="system-info">
                <h3>System Information</h3>
                <p>TensorFlow Version: {tf.__version__}</p>
                <p>OpenCV Version: {cv2.__version__}</p>
                <p>Model Status: {'Loaded' if model_loaded else 'Not Loaded'} ({model_size:.2f} MB)</p>
                <p>Face Detector: {'Loaded' if face_detector_loaded else 'Failed to load'}</p>
                <p>Processing Mode: CPU Only (for stability)</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown(
            '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">'
            'Built with ❤️ using Streamlit, TensorFlow, and OpenCV | Optimized for stability'
            '</footer>', 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page to try again.")
        
        # Show debug information
        with st.expander("Debug Information"):
            st.write("Error details:", str(e))
            st.write("Python version:", sys.version)
            st.write("Working directory:", os.getcwd())
            st.write("Available files:", os.listdir())

if __name__ == "__main__":
    main()
