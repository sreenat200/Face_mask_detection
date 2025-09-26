import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import os
import sys
import requests

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
        
        /* Image container */
        .image-container {
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

# Force TensorFlow to use CPU only (Streamlit Cloud has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False

def download_file(url, dest_path):
    """Download a file from a URL to the specified destination."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download file from {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

def load_model():
    """Load the Keras face mask detection model with enhanced error handling."""
    global model, model_loaded
    if model is None:
        model_path = "mask_detection_model.h5"
        
        # Check if model file exists, attempt to download if not present
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}. Attempting to download...")
            # Replace with actual URL to your model file (e.g., hosted on GitHub or a cloud storage)
            model_url = "https://your-model-hosting-url/mask_detection_model.h5"
            if not download_file(model_url, model_path):
                st.error("Failed to download model file. Please ensure the model is available.")
                model_loaded = False
                return None
        
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model_loaded = True
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            model_loaded = False
            return None
    return model

def load_face_detector():
    """Load OpenCV's Haar cascade for face detection."""
    global face_cascade, face_detector_loaded
    if face_cascade is None:
        try:
            # Load the pre-trained Haar cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                st.warning("Haar cascade file not found locally. Attempting to download...")
                cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                cascade_dir = "haarcascades"
                os.makedirs(cascade_dir, exist_ok=True)
                cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
                if not download_file(cascade_url, cascade_path):
                    st.error("Failed to download Haar cascade file.")
                    face_detector_loaded = False
                    return False
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                st.error("Failed to load face detector. Ensure OpenCV is installed correctly.")
                face_detector_loaded = False
                return False
            
            face_detector_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading face detector: {str(e)}")
            face_detector_loaded = False
            return False
    return True

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    resized = cv2.resize(image, model_input_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces in the image using Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return [(x, y, w, h) for (x, y, w, h) in faces]

def classify_faces(image: np.ndarray, faces: list[tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> list[dict]:
    """Classify each detected face as mask or no mask."""
    detections = []
    
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        processed_face = preprocess_image(face_roi)
        try:
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

def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Use default font to avoid file access issues in Streamlit Cloud
    font = ImageFont.load_default()
    
    colors = {
        "Mask": (0, 255, 0),      # Green
        "No Mask": (255, 0, 0),   # Red
    }
    
    for detection in detections:
        try:
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
        except Exception as e:
            st.warning(f"Error drawing detection: {str(e)}")
    
    return np.array(pil_image)

def main():
    """Main function to run the Streamlit app."""
    try:
        # Header
        st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="description">Upload an image to detect faces and classify whether they are wearing a mask or not.</p>', unsafe_allow_html=True)
        
        # Load model and face detector
        model = load_model()
        load_face_detector()
        
        # Check if models loaded successfully
        if not model_loaded or not face_detector_loaded:
            st.error("Failed to load the model or face detector. Please check the files and try again.")
            
            # Debugging information
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">🔍 Debugging Information</h3>', unsafe_allow_html=True)
            st.write("**Current Directory:**", os.getcwd())
            st.write("**Files in Directory:**")
            for file in os.listdir():
                if file.endswith(('.h5', '.keras')):
                    st.write(f"- {file}")
            st.write("**System Information:**")
            st.write(f"- Python Version: {sys.version}")
            st.write(f"- TensorFlow Version: {tf.__version__}")
            st.write(f"- OpenCV Version: {cv2.__version__}")
            
            model_path = "mask_detection_model.h5"
            if os.path.exists(model_path):
                st.write(f"\n**Model File Information:**")
                st.write(f"- File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
                st.write(f"- File exists: Yes")
            
            return
        
        # Sidebar
        with st.sidebar:
            st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
            
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
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            
            # Image upload
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    # Read and process the image
                    image = Image.open(uploaded_file).convert("RGB")
                    img_array = np.array(image)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Detect faces
                    faces = detect_faces(img_array)
                    
                    # Classify faces
                    detections = classify_faces(img_array, faces, confidence_threshold)
                    
                    # Draw detections
                    annotated_img = draw_detections(img_array, detections)
                    
                    # Convert back to RGB for display
                    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, caption="Processed Image", use_column_width=True)
                    
                    # Display detection results
                    if detections:
                        st.write("**Detections:**")
                        for detection in detections:
                            st.write(f"- {detection['label']} (Confidence: {detection['score']:.2%})")
                    else:
                        st.write("No faces detected or confidence below threshold.")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Instructions
            st.info("""
                **Instructions:**
                1. Upload an image using the file uploader.
                2. The system will detect faces and classify mask usage.
                3. Green boxes = With mask, Red boxes = Without mask.
            """)
        
        with col2:
            st.markdown('<h3 class="sidebar-title">🎯 Detection Legend</h3>', unsafe_allow_html=True)
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
        
        # System information
        st.markdown("---")
        st.markdown("""
            <div class="system-info">
                <h3>System Information</h3>
                <p>TensorFlow Version: {tf_version}</p>
                <p>Model: {model_name} ({model_size:.2f} MB)</p>
                <p>Face Detector: {detector_status}</p>
            </div>
        """.format(
            tf_version=tf.__version__,
            model_name="mask_detection_model.h5",
            model_size=os.path.getsize("mask_detection_model.h5") / (1024*1024) if os.path.exists("mask_detection_model.h5") else 0,
            detector_status="Loaded" if face_detector_loaded else "Failed to load"
        ), unsafe_allow_html=True)
        
        # Footer
        st.markdown(
            '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">'
            'Built with ❤️ using Streamlit, TensorFlow, and OpenCV'
            '</footer>', 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
