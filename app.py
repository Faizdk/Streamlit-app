import streamlit as st
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import shutil

# Folder konfigurasi
UPLOAD_FOLDER = "static/uploads"
PREDICT_FOLDER = "runs/detect/predict"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLOv8
st.sidebar.title("🔧 Model Info")
model_path = "best.pt"
model = YOLO(model_path)
st.sidebar.success(f"Model loaded: {model_path}")

# Judul utama
st.title("🚗 Vehicle Class Detection with YOLOv8")

# Upload file
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file
    image_id = str(uuid.uuid4())
    saved_filename = image_id + ".jpg"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar asli
    st.subheader("Original Image")
    st.image(saved_path, use_container_width=True)

    # Bersihkan folder predict sebelumnya
    if os.path.exists(PREDICT_FOLDER):
        shutil.rmtree(PREDICT_FOLDER)

        # Bersihkan folder predict sebelumnya
    if os.path.exists(PREDICT_FOLDER):
        shutil.rmtree(PREDICT_FOLDER)

    # Prediksi
    with st.spinner("Running object detection..."):
        results = model(saved_path, save=True)
        result = results[0]

    # Path ke hasil prediksi
    result_img_path = os.path.join(result.save_dir, saved_filename)

    # Tampilkan hasil deteksi gambar
    st.subheader("🔍 Detected Image")
    st.image(result_img_path, use_container_width=True)

    # Ekstrak data deteksi ke tabel
    if result.boxes is not None:
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # Daftar label dari model
        class_names = result.names

        # Data tabel
        table_data = []
        detected_labels = set()

        for i in range(len(class_ids)):
            label = class_names[class_ids[i]]
            confidence = round(float(scores[i]), 2)
            x1, y1, x2, y2 = [int(coord) for coord in xyxy[i]]

            table_data.append({
                "Class": label,
                "Confidence": confidence,
                "Xmin": x1,
                "Ymin": y1,
                "Xmax": x2,
                "Ymax": y2
            })

            detected_labels.add(label)

        st.subheader("📊 Detection Table")
        st.dataframe(table_data)

        st.subheader("🚘 Detected Vehicle Classes")
        st.write(", ".join(sorted(detected_labels)))
    else:
        st.warning("Tidak ada objek terdeteksi.")

    # Tombol download hasil
    with open(result_img_path, "rb") as result_file:
        st.download_button(
            label="📥 Download result",
            data=result_file,
            file_name="detected_" + saved_filename,
            mime="image/jpeg"
        )

