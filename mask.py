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
from huggingface_hub import hf_hub_download  # For downloading from Hugging Face

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

# 🔒 Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
tf.config.set_visible_devices([], 'GPU')   # Explicitly disable GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Global variables for model and processor
model = None
face_cascade = None
model_input_size = (128, 128)  # From model config
class_names = ['Mask', 'No Mask']  # From model config
model_loaded = False
face_detector_loaded = False


def load_model():
    """Load the Keras face mask detection model from Hugging Face Hub."""
    global model, model_loaded
    if model is not None:
        return model

    try:
        repo_id = "sreenathsree1578/face_mask_detection"
        filename = "mask_detection_model.h5"

        st.info(f"📥 Downloading model from Hugging Face: {repo_id}/{filename}")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        st.success(f"✅ Model downloaded: {model_path}")

        # Load model without compiling (safe loading)
        model = tf.keras.models.load_model(model_path, compile=False)
        model_loaded = True

        st.success("🧠 Face mask detection model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Failed to load model from Hugging Face: {str(e)}")
        st.warning("Ensure you're connected to the internet and the repository exists.")
        model_loaded = False
        return None


def load_face_detector():
    """Load OpenCV's Haar cascade for face detection."""
    global face_cascade, face_detector_loaded
    if face_cascade is not None:
        return True

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("⚠️ Failed to load Haar Cascade file. Check OpenCV installation.")
            face_detector_loaded = False
            return False
        face_detector_loaded = True
        return True
    except Exception as e:
        st.error(f"⚠️ Error loading face detector: {str(e)}")
        face_detector_loaded = False
        return False


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for model inference."""
    resized = cv2.resize(image, model_input_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
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


def classify_faces(image: np.ndarray, faces: List[Tuple[int, int, int, int]], confidence_threshold: float = 0.5) -> List[Dict]:
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
            st.warning(f"⚠️ Classification error: {str(e)}")
    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = {
        "Mask": (0, 255, 0),
        "No Mask": (255, 0, 0),
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
            st.warning(f"🎨 Drawing error: {str(e)}")

    return np.array(pil_image)


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
            st.error(f"🎥 Frame processing error: {str(e)}")
            return frame

    def get_average_fps(self) -> float:
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


def main():
    """Main function to run the Streamlit app."""
    try:
        st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="description">Real-time face mask detection using a Keras model from Hugging Face. Detects whether people are wearing masks.</p>', unsafe_allow_html=True)

        # Load model from Hugging Face
        model = load_model()
        load_face_detector()

        if not model_loaded or not face_detector_loaded:
            st.error("🛑 Failed to load the model or face detector.")
            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">🔍 Debug Info</h3>', unsafe_allow_html=True)

            st.write("**Python Version:**", sys.version)
            st.write("**TensorFlow Version:**", tf.__version__)
            st.write("**OpenCV Version:**", cv2.__version__)

            st.info("""
                Make sure:
                - You have an internet connection.
                - The repository [sreenathsree1578/face_mask_detection](https://huggingface.co/sreenathsree1578/face_mask_detection) exists.
                - Required package: `pip install huggingface_hub`
            """)
            return

        # Sidebar Settings
        with st.sidebar:
            st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)

            video_size = st.selectbox(
                "Video Size",
                options=["640x480", "1280x720", "1920x1080"],
                index=0
            )

            fps = st.slider("Frames Per Second (FPS)", 5, 30, 15)

            mirror_video = st.checkbox("Mirror Video", False)

            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.1, 0.9, 0.5, 0.05
            )

            st.markdown("---")
            st.markdown('<h3 class="sidebar-title">🔍 Face Detection</h3>', unsafe_allow_html=True)

            scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01)
            min_neighbors = st.slider("Min Neighbors", 1, 10, 5, 1)

            # Update cascade parameters
            if face_cascade:
                # Note: We can't dynamically update OpenCV cascade settings during runtime easily
                pass

        # Parse size
        width, height = map(int, video_size.split('x'))

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)

            webrtc_ctx = webrtc_streamer(
                key="face-mask-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: FaceMaskProcessor(
                    target_size=(width, height),
                    confidence_threshold=confidence_threshold,
                    mirror=mirror_video
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

            st.info("""
                **Instructions:**
                1. Click "START" to begin.
                2. Allow camera access.
                3. Green = With Mask, Red = Without Mask.
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

        # System Info
        st.markdown("---")
        model_size_mb = 0
        model_path = None
        try:
            from huggingface_hub import cached_assets_path
            cache_dir = cached_assets_path(library_name="huggingface_hub")
            # This is approximate; actual path depends on HF cache
            import glob
            files = glob.glob(f"{cache_dir}/**/mask_detection_model.h5", recursive=True)
            if files:
                model_path = files[0]
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        except:
            model_size_mb = "Unknown"

        st.markdown(f"""
            <div class="system-info">
                <h3>System Information</h3>
                <p><strong>Framework:</strong> TensorFlow {tf.__version__}</p>
                <p><strong>Model Source:</strong> Hugging Face (sreenathsree1578/face_mask_detection)</p>
                <p><strong>Model Size:</strong> {model_size_mb if isinstance(model_size_mb, str) else f'{model_size_mb:.2f}'} MB</p>
                <p><strong>Device:</strong> CPU Only (GPU disabled)</p>
                <p><strong>Status:</strong> Model {'Loaded' if model_loaded else 'Failed'}, Detector {'Loaded' if face_detector_loaded else 'Failed'}</p>
            </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown(
            '<footer style="text-align: center; color: #b0b0b0; font-size: 0.9rem;">'
            'Built with ❤️ using Streamlit, TensorFlow, OpenCV & Hugging Face'
            '</footer>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"💥 An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
