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
import h5py  # Required to read .h5 file internals

# Set page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        .main { background-color: #121212; }
        .css-1d391kg { background-color: #1e1e1e; }
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
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -2px rgba(0,0,0,0.2);
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
        .legend-card {
            background-color: #2a2a2a;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        footer {
            color: #b0b0b0;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #333333;
            text-align: center;
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
        .system-info h3 {
            color: #aaa;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🔒 Force CPU Only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Global variables
model = None
face_cascade = None
model_input_size = (128, 128)
class_names = ['Mask', 'No Mask']
model_loaded = False
face_detector_loaded = False


def load_model():
    """Load and fix the Keras model from Hugging Face that has batch_shape issues."""
    global model, model_loaded
    if model is not None:
        return model

    try:
        from huggingface_hub import hf_hub_download

        repo_id = "sreenathsree1578/face_mask_detection"
        filename = "mask_detection_model.h5"

        st.info(f"📥 Downloading model: {repo_id}/{filename}")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        st.success(f"✅ Model downloaded: {model_path}")

        # --- Fix the model config to remove 'batch_shape' ---
        def fix_model_config(model_path):
            with h5py.File(model_path, 'r+') as f:
                # Read model config
                model_config_str = f.attrs['model_config'].decode('utf-8')
                import json
                config = json.loads(model_config_str)

                # Recursive function to replace batch_shape with input_shape
                def traverse(config):
                    if isinstance(config, dict):
                        if config.get("class_name") == "InputLayer":
                            if "config" in config:
                                if "batch_shape" in config["config"]:
                                    del config["config"]["batch_shape"]
                                config["config"]["input_shape"] = (128, 128, 3)
                        for k in config:
                            config[k] = traverse(config[k])
                    elif isinstance(config, list):
                        return [traverse(x) for x in config]
                    return config

                fixed_config = traverse(config)
                f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')

        # Apply fix directly to the file
        fix_model_config(model_path)

        # Now safely load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        model_loaded = True
        st.success("🧠 Model loaded successfully after fixing InputLayer!")
        return model

    except Exception as e:
        st.error(f"❌ Failed to load or fix model: {str(e)}")
        st.exception(e)
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
            st.error("⚠️ Failed to load Haar Cascade.")
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
    """Detect faces using Haar cascade."""
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
    """Classify each face as mask or no mask."""
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
                    "box": {"xmin": x, "ymin": y, "ymax": y + h, "xmax": x + w}
                })
        except Exception as e:
            st.warning(f"⚠️ Classification error: {str(e)}")
    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = {"Mask": (0, 255, 0), "No Mask": (255, 0, 0)}

    for det in detections:
        box = det["box"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        label = det["label"]
        conf = det["score"]
        color = colors.get(label, (0, 0, 255))

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)

        text = f"{label}: {conf:.2%}"
        bbox = draw.textbbox((0, 0), text, font=font)
        txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle([(xmin, ymin - txt_h - 6), (xmin + txt_w + 10, ymin - 6)], fill=color)
        draw.text((xmin + 5, ymin - txt_h - 6), text, fill="white", font=font)

    return np.array(pil_image)


class FaceMaskProcessor(VideoProcessorBase):
    """Real-time video processor."""

    def __init__(self, target_size: Tuple[int, int] = (640, 480), confidence_threshold: float = 0.5, mirror: bool = False):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.mirror = mirror
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.mirror:
            img = cv2.flip(img, 1)

        if img.shape[:2][::-1] != self.target_size:
            img = cv2.resize(img, self.target_size)

        try:
            faces = detect_faces(img)
            detections = classify_faces(img, faces, self.confidence_threshold)
            annotated_img = draw_detections(img, detections)
        except Exception as e:
            annotated_img = img  # fallback
            st.warning(f"Error in processing: {e}")

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")


def main():
    st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Real-time mask detection using a model from Hugging Face — now fixed for compatibility.</p>', unsafe_allow_html=True)

    # Load model and detector
    model = load_model()
    load_face_detector()

    if not model_loaded or not face_detector_loaded:
        st.error("🛑 Failed to load model or face detector.")
        st.markdown("---")
        st.markdown('<h3 class="sidebar-title">🔍 Debug Info</h3>', unsafe_allow_html=True)
        st.write("**TensorFlow Version:**", tf.__version__)
        st.write("**OpenCV Version:**", cv2.__version__)
        st.info("Ensure internet access and correct model repository.")
        return

    # Sidebar settings
    with st.sidebar:
        st.markdown('<h3 class="sidebar-title">🎛️ Settings</h3>', unsafe_allow_html=True)
        video_size = st.selectbox("Video Size", ["640x480", "1280x720"], index=0)
        fps = st.slider("FPS", 5, 30, 15)
        mirror = st.checkbox("Mirror Video", False)
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

    width, height = map(int, video_size.split('x'))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        webrtc_ctx = webrtc_streamer(
            key="face-mask-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: FaceMaskProcessor(
                target_size=(width, height),
                confidence_threshold=confidence,
                mirror=mirror
            ),
            media_stream_constraints={
                "video": {"width": {"ideal": width}, "height": {"ideal": height}, "frameRate": {"ideal": fps}},
                "audio": False
            },
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("🟢 Click START and allow camera access.")

    with col2:
        st.markdown('<h3 class="sidebar-title">🎯 Legend</h3>', unsafe_allow_html=True)
        st.markdown("""
            <div class="legend-card" style="border-color: #22c55e;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #22c55e; margin-right: 10px;"></div>
                    <strong>Mask</strong>
                </div>
            </div>
            <div class="legend-card" style="border-color: #ef4444;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; background-color: #ef4444; margin-right: 10px;"></div>
                    <strong>No Mask</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # System info
    st.markdown("---")
    st.markdown("""
        <div class="system-info">
            <h3>System Info</h3>
            <p><strong>Status:</strong> Model Loaded ✅ | Detector Ready ✅</p>
            <p><strong>Device:</strong> CPU Only</p>
            <p><strong>Framework:</strong> TensorFlow {}</p>
        </div>
    """.format(tf.__version__), unsafe_allow_html=True)

    st.markdown(
        '<footer>🔐 Built with ❤️ using Streamlit, TF, OpenCV & Hugging Face</footer>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
