import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError, Accuracy
from collections import deque
from io import BytesIO
import os

# Face Mask Detection Model
class MaskDetectionCNN(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes=2, in_channels=3):
        super(MaskDetectionCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_transform(in_channels):
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels)
    ])

# Define labels for mask detection
mask_labels = ['Mask', 'No Mask']

st.markdown("<h3>Live Face Mask Detection</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["Model 1", "Model 2"],
        index=0
    )
    mode = st.selectbox("Select Mode", ["Video Mode", "Snap Mode"], index=0)
    if mode == "Video Mode":
        quality = st.selectbox("Select Video Quality", ["Low (480p)", "Medium (720p)", "High (1080p)"], index=1)
        fps = st.selectbox("Select FPS", [15, 30, 60], index=1)
        mirror_feed = st.checkbox("Mirror Video Feed", value=True)
    else:
        mirror_snap = st.checkbox("Mirror Snap Image", value=True)

quality_map = {
    "High (1080p)": {"width": 1920, "height": 1080},
    "Low (480p)": {"width": 854, "height": 480},
    "Medium (720p)": {"width": 1280, "height": 720}
}

@st.cache_resource
def load_mask_detection_model():
    try:
        config_path = hf_hub_download(repo_id="sreenathsree1578/face_mask_detection", filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config.get("num_classes", 2)
        model = MaskDetectionCNN(num_classes=num_classes, in_channels=3)
        model = model.from_pretrained("sreenathsree1578/face_mask_detection")
        model.eval()
        return model, 3
    except Exception as e:
        with st.sidebar:
            st.warning(f"Error loading face_mask_detection: {str(e)}. Using default.")
        return MaskDetectionCNN(num_classes=2, in_channels=3), 3

# Load model
model, in_channels = load_mask_detection_model()
transform_live = get_transform(in_channels)

mask_colors = {
    'Mask': (0, 255, 0),      # Green
    'No Mask': (255, 0, 0)    # Red
}

def process_single_image(img, mirror=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if mirror:
        img = cv2.flip(img, 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    mask_status = None
    
    if len(faces) == 0:
        return img, mask_status

    for (x, y, w, h) in faces:
        # Mask detection
        face_mask = img[y:y+h, x:x+w]
        face_mask = cv2.resize(face_mask, (48, 48))
        face_mask_rgb = cv2.cvtColor(face_mask, cv2.COLOR_BGR2RGB)
        face_mask_pil = Image.fromarray(face_mask_rgb, mode='RGB')
        face_mask_tensor = transform_live(face_mask_pil).unsqueeze(0)
        try:
            with torch.no_grad():
                output_mask = model(face_mask_tensor)
                _, pred_mask = torch.max(output_mask, 1)
                mask_status = mask_labels[pred_mask.item()] if pred_mask.item() < len(mask_labels) else "unknown"
        except Exception as e:
            with st.sidebar:
                st.warning(f"Mask prediction failed: {str(e)}. Input shape: {face_mask_tensor.shape}")
            mask_status = "unknown"

        # Draw annotations on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        mask_color = mask_colors.get(mask_status, (255, 0, 0))
        text_size_mask = cv2.getTextSize(mask_status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x, y-30), (x+text_size_mask[0], y), (255, 255, 255), -1)
        cv2.putText(img, mask_status, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_color, 2)

    return img, mask_status

if mode == "Video Mode":
    resolution = quality_map[quality]

    class MaskProcessor(VideoProcessorBase):
        def __init__(self, mirror=False):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.mirror = mirror
            self.no_face_count = 0
            self.frame_count = 0
            self.last_mask_status = "unknown"

        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            if self.mirror:
                img = cv2.flip(img, 1)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                self.no_face_count += 1
                if self.no_face_count % 30 == 0:
                    with st.sidebar:
                        st.warning("No faces detected in the frame.")
                mask_status = self.last_mask_status
            else:
                self.no_face_count = 0
                for (x, y, w, h) in faces:
                    if self.frame_count % 3 == 0:
                        face_mask = img[y:y+h, x:x+w]
                        face_mask = cv2.resize(face_mask, (48, 48))
                        face_mask_rgb = cv2.cvtColor(face_mask, cv2.COLOR_BGR2RGB)
                        face_mask_pil = Image.fromarray(face_mask_rgb, mode='RGB')
                        face_mask_tensor = transform_live(face_mask_pil).unsqueeze(0)
                        try:
                            with torch.no_grad():
                                output_mask = model(face_mask_tensor)
                                _, pred_mask = torch.max(output_mask, 1)
                                mask_status = mask_labels[pred_mask.item()] if pred_mask.item() < len(mask_labels) else "unknown"
                        except Exception as e:
                            with st.sidebar:
                                st.warning(f"Mask prediction failed: {str(e)}. Input shape: {face_mask_tensor.shape}")
                            mask_status = "unknown"

                        self.last_mask_status = mask_status
                    else:
                        mask_status = self.last_mask_status

                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    mask_color = mask_colors.get(mask_status, (255, 0, 0))
                    text_size_mask = cv2.getTextSize(mask_status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img, (x, y-30), (x+text_size_mask[0], y), (255, 255, 255), -1)
                    cv2.putText(img, mask_status, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mask_color, 2)

            return frame.from_ndarray(img, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    try:
        webrtc_streamer(
            key="mask-detection",
            video_processor_factory=lambda: MaskProcessor(mirror=mirror_feed),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": resolution["width"]},
                    "height": {"ideal": resolution["height"]},
                    "frameRate": {"ideal": fps},
                    "deviceId": {"exact": 1}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_config
        )
    except Exception as e:
        with st.sidebar:
            if "OverconstrainedError" in str(e):
                st.warning("Please select a device to continue.")
            else:
                st.warning(f"Camera 1 failed: {str(e)}. Switching to camera 0.")
                try:
                    webrtc_streamer(
                        key="mask-detection-fallback",
                        video_processor_factory=lambda: MaskProcessor(mirror=mirror_feed),
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": resolution["width"]},
                                "height": {"ideal": resolution["height"]},
                                "frameRate": {"ideal": fps},
                                "deviceId": {"exact": 0}
                            },
                            "audio": False
                        },
                        async_processing=True,
                        rtc_configuration=rtc_config
                    )
                except Exception as e2:
                    if "OverconstrainedError" in str(e2):
                        st.warning("Please select a device to continue.")
                    else:
                        st.error(f"Camera 0 failed: {str(e2)}.")
else:
    st.header("Snap Mode")
    if mirror_snap:
        st.markdown(
            """
            <style>
            video {
                transform: scaleX(-1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    image = st.camera_input("Take a photo")
    if image is not None:
        image_pil = Image.open(BytesIO(image.getvalue()))
        img_rgb = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        processed_img, mask_status = process_single_image(img_bgr, mirror=mirror_snap)
        
        if mask_status is None:
            with st.sidebar:
                st.warning("No faces detected in the photo.")
            st.markdown("<h4 style='color: red; text-align: center;'>No face detected in the photo.</h4>", unsafe_allow_html=True)
            st.image(processed_img, channels="BGR", caption="Processed Image")
        else:
            # Display results in a consistent format
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Detection Results")
                
                # Mask status display
                mask_rgb = mask_colors.get(mask_status, (255, 0, 0))
                mask_display = mask_status if mask_status is not None else "unknown"
                st.markdown(f"**Mask Status**: <span style='color: #{mask_rgb[0]:02x}{mask_rgb[1]:02x}{mask_rgb[2]:02x}; font-size: 20px'>{mask_display}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.image(processed_img, channels="BGR", caption="Processed Image")
