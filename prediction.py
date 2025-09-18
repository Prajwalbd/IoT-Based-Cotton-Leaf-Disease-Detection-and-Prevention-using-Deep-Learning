
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.applications import MobileNetV2
import pandas as pd

model = tf.keras.models.load_model(' ')  # Load trained model
feature_extractor = MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg')

class_labels = ['Bacterial_Blight', 'Fussarium_wilt', 'Healthy', 'red_leaf'] # Alter as per required
disease_solutions = {
    'Bacterial_Blight': "- Spray **Copper Oxychloride (2.5 g/l)** every 10–15 days.\n"
    "- For severe cases, combine with **Streptocycline (0.1 g/l)**.\n"
    "- Use certified seeds and avoid overhead irrigation.\n"
    "- Remove infected plant parts and practice crop rotation.\n"
    "- Use resistant cotton varieties if available.",
    'red_leaf': "- Spray **Imidacloprid (0.3 ml/l)** or **Thiamethoxam (0.25 g/l)**.\n"
    "- Use **Acetamiprid** or **Spiromesifen** for better results.\n"
    "- Install yellow sticky traps to reduce whitefly population.\n"
    "- Remove infected plants and avoid susceptible crops like okra nearby.\n"
    "- Use virus-resistant cotton varieties if available.",
    'Fussarium_wilt':  "- Treat seeds with **Carbendazim (2–3 g/kg)** before sowing.\n"
    "- Apply **Trichoderma viride** as a biological control agent.\n"
    "- Improve drainage and avoid overwatering.\n"
    "- Practice **crop rotation** with non-host crops (e.g., cereals).\n"
    "- Remove infected plants early and avoid excess nitrogen fertilizers.\n"
    "- Use resistant cotton varieties where available.",
    'Healthy': "No disease detected. Keep monitoring and maintain good agricultural practices."
}

def send_to_esp32(disease, accuracy, remedies, esp32_ip=''):
    url = f"{esp32_ip}/update"    #ESP Url to connect the hardware  
    payload = {
        "disease": disease,
        "accuracy": accuracy,  
        "remedies": remedies
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            st.success("✅ Data sent to ESP32 successfully!")
        else:
            st.error(f"❌ ESP32 responded with: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Failed to send data to ESP32: {e}")



def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def contains_leaf(image, green_threshold=0.6):
    img_cv = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    return green_ratio >= green_threshold



def predict_image(image, threshold=0.6):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)

    if max_prob < threshold:
        return "Unknown Leaf", predictions[0]

    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class, predictions[0]
def extract_features(image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    return features.flatten()


def show_prediction(image):
    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # Check if the image likely contains a leaf
    if not contains_leaf(image):
        st.warning("⚠ This image does not appear to contain a cotton leaf. Please upload or capture a proper leaf image.")
        return

    label, probs = predict_image(image)

    if label == "Unknown Leaf":
        st.warning("❓ Unable to confidently classify this leaf.")
        return

    confidence = probs[class_labels.index(label)]
    st.subheader(f"🧠 Prediction: **{label.upper()}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    # Show remedy
    remedy_text = disease_solutions.get(label, "No remedy available.")
    st.subheader("💊 Recommended Solution")
    st.info(remedy_text)

    # Send to ESP32
    send_to_esp32(label, f"{confidence * 100:.2f}%", remedy_text)

    # Features (optional)
    features = extract_features(image)
    feature_names = [
        "Low-level Feature 1 (Color)",
        "Low-level Feature 2 (Texture)",
        "Low-level Feature 3 (Edge Detection)",
        "High-level Feature 1 (Shape)",
        "High-level Feature 2 (Pattern)",
        "High-level Feature 3 (Details)",
    ]
    feature_data = pd.DataFrame({
        'Feature Name': feature_names[:6],
        'Feature Value': features[:6]
    })
    st.write("Here are the first 6 features:")
    st.table(feature_data)


def capture_from_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        st.error("❌ Unable to access camera.")
        return
    st.info("📸 Capturing image in 3 seconds...")
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("❌ Failed to capture image.")
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    show_prediction(image)



def capture_from_ip_webcam(url=''):
    try:
        st.info("📡 Capturing image from IP Webcam...")
        time.sleep(2)

        response = requests.get(url, timeout=5)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        show_prediction(image)

    except Exception as e:
        st.error(f"❌ Error: {e}")

    
st.title("🍃 Cotton Leaf Disease Detection")
st.write("Upload or capture a cotton leaf image to detect its health condition.")

col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 Device Camera"):
        capture_from_camera(0)
with col2:
    if st.button("🌐 IP Webcam"):
        capture_from_ip_webcam()

st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload a cotton leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    show_prediction(image)

