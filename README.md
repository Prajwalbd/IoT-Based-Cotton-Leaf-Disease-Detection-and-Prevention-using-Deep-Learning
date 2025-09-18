# IoT-Based-Cotton-Leaf-Disease-Detection-and-Prevention-using-Deep-Learning
 Cotton leaf disease detection with deep learning involves training CNNs on labeled leaf images to identify  diseases. The model, fine-tuned for accuracy, is deployed in a real-time application to diagnose and  recommend treatments. Continuous improvement is achieved by updating the model with new data.
 This project combines Deep Learning (CNN) and IoT (ESP32 + Sprayer System) to provide an automated solution for detecting cotton leaf diseases and applying targeted pesticide treatment. It enhances crop yield, reduces labor, and minimizes pesticide overuse.
# Features
-Detects multiple cotton leaf diseases using CNN (MobileNetV2).
-Real-time results with confidence scores.
-Automated spraying system triggered by ESP32 + Relay + Motor.
-User-friendly Streamlit web interface for farmers.
-Scalable for open fields, greenhouses, and research labs.
# Tech Stack
-Software: Python 3.12, TensorFlow/Keras, OpenCV, Streamlit, NumPy
-Hardware: ESP32 Microcontroller, Relay Module, DC Motor + Pump, Pesticide Sprayer
# Workflow
1.Capture/upload cotton leaf image via Streamlit.
2.CNN model classifies leaf → Healthy / Diseased.
3.If diseased → ESP32 triggers relay-controlled sprayer.
4.Farmers view results and remedies instantly on the web UI.
