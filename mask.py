import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import time
from typing import Tuple, List, Dict, Any
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

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
        
        /* Model info */
        .model-info {
            background-color: #2a2a2a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #444;
        }
    </style>
""", unsafe_allow_html=True)

# Force TensorFlow to use CPU only and disable optimizations that cause segfaults
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False

def load_model():
    """Load the face mask detection model from Hugging Face Hub."""
    global model, model_loaded
    if model is None:
        try:
            # Import transformers inside the function to avoid global import issues
            from transformers import pipeline
            
            # Load the model from Hugging Face Hub
            model = pipeline(
                "object-detection",
                model="sreenathsree1578/face_mask_detection",
                device=-1  # Use CPU
            )
            model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error loading model from Hugging Face: {str(e)}")
            st.info("Make sure you have an internet connection and the model is accessible.")
            model_loaded = False
            return None
    return model

def load_face_detector():
    """Load OpenCV's Haar cascade for face detection (fallback)."""
    global face_cascade, face_detector_loaded
    if face_cascade is None:
        try:
            # Load the pre-trained Haar cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Check if the cascade was loaded successfully
            if face_cascade.empty():
                face_detector_loaded = False
                return False
            
            face_detector_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading face detector: {str(e)}")
            face_detector_loaded = False
            return False
    return True

def detect_faces_and_masks(image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """Detect faces and classify mask usage using the Hugging Face model."""
    detections = []
    
    try:
        # Convert BGR to RGB for the model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Run inference
        results = model(pil_image)
        
        # Process results
        for result in results:
            box = result['box']
            score = result['score']
            label = result['label']
            
            # Only include detections above confidence threshold
            if score >= confidence_threshold:
                detections.append({
                    "label": label,
                    "score": score,
                    "box": {
                        "xmin": box['xmin'],
                        "ymin": box['ymin'],
                        "xmax": box['xmax'],
                        "ymax": box['ymax']
                    }
                })
                
    except Exception as e:
        st.warning(f"Error during detection: {str(e)}")
    
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
        "with_mask": (0, 255, 0),      # Green
        "without_mask": (255, 0, 0),   # Red
        "mask_weared_incorrect": (255, 165, 0)  # Orange
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
            label_text = f"{label.replace('_', ' ').title()}: {confidence:.2%}"
            
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
        
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror the image if requested
            if self.mirror:
                img = cv2.flip(img, 1)
            
            # Resize frame if needed
            if img.shape[:2][::-1] != self.target_size:
                img = cv2.resize(img, self.target_size)
            
            # Detect faces and classify masks
            detections = detect_faces_and_masks(img, self.confidence_threshold)
            
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
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            # Return original frame if processing fails
            return frame
    
    def get_average_fps(self) -> float:
        """Calculate average FPS based on processing times."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def main():
    """Main function to run the Streamlit app."""
    try:
        # Header
        st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="description">Real-time face mask detection using a model from Hugging Face. The system detects faces and classifies mask usage in real-time.</p>', unsafe_allow_html=True)
        
        # Model information
        st.markdown("""
            <div class="model-info">
                <h3>🤖 Model Information</h3>
                <p><strong>Model:</strong> sreenathsree1578/face_mask_detection</p>
                <p><strong>Source:</strong> Hugging Face Hub</p>
                <p><strong>Type:</strong> Object Detection</p>
                <p><strong>Classes:</strong> with_mask, without_mask, mask_weared_incorrect</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load model from Hugging Face
        model = load_model()
        
        # Check if model loaded successfully
        if not model_loaded:
            st.error("Failed to load the model from Hugging Face. Please check your internet connection and try again.")
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
                4. Green boxes = With mask, Red boxes = Without mask, Orange boxes = Incorrect mask usage
            """)
        
        with col2:
            st.markdown('<h3 class="sidebar-title">🎯 Detection Legend</h3>', unsafe_allow_html=True)
            
            # Create legend cards
            st.markdown("""
                <div class="legend-card" style="border-color: #22c55e;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px; border-radius: 4px;"></div>
                        <strong>With Mask</strong>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is wearing a mask correctly</p>
                </div>
                
                <div class="legend-card" style="border-color: #ef4444;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px; border-radius: 4px;"></div>
                        <strong>Without Mask</strong>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Person is not wearing a mask</p>
                </div>
                
                <div class="legend-card" style="border-color: #f97316;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 20px; height: 20px; background-color: #f97316; margin-right: 10px; border-radius: 4px;"></div>
                        <strong>Incorrect Usage</strong>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #b0b0b0; font-size: 0.9rem;">Mask is worn incorrectly</p>
                </div>
            """, unsafe_allow_html=True)
        
        # System information at the bottom
        st.markdown("---")
        st.markdown("""
            <div class="system-info">
                <h3>System Information</h3>
                <p>Model: sreenathsree1578/face_mask_detection (Hugging Face)</p>
                <p>Model Status: {model_status}</p>
            </div>
        """.format(
            model_status="Loaded" if model_loaded else "Failed to load"
        ), unsafe_allow_html=True)
        
        # Footer
        st.markdown(
            '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">'
            'Built with ❤️ using Streamlit, Hugging Face Transformers, and OpenCV'
            '</footer>', 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
