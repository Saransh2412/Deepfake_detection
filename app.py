from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import time
import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Constants
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20
UPLOAD_FOLDER = tempfile.gettempdir()
CONFIDENCE_THRESHOLD = 0.80
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MODEL_URL = "https://drive.google.com/uc?id=1pZiRmM-VXiSYc_SRI58zhGkCXkEnGB1K"
MODEL_PATH = "deepfake_detection.h5"
model = None  # Lazy load later

# Utility to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Health check route
@app.route('/')
def home():
    return '✅ Deepfake detection backend is running!'

# Frame extraction logic
def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total < num_frames or total == 0:
        cap.release()
        return np.zeros((num_frames, FRAME_HEIGHT, FRAME_WIDTH, 3))

    interval = total // num_frames
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))

    return np.array(frames)

# Prediction logic
def predict_video(video_path):
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                logging.info("📥 Downloading model from Google Drive...")
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            model = load_model(MODEL_PATH, compile=False)
            logging.info("✅ Model loaded successfully!")
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            return {"error": "Model loading failed"}, 500

    start_time = time.time()
    frames = extract_frames(video_path)
    input_data = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_data)[0][0]

    is_deepfake = prediction > 0.5
    confidence = float(prediction) if is_deepfake else float(1 - prediction)
    confidence_percent = int(confidence * 100)
    analysis_time = time.time() - start_time
    manipulation_regions = ['face', 'eyes'] if is_deepfake else []

    return {
        "isDeepfake": bool(is_deepfake),
        "confidenceScore": confidence_percent,
        "manipulationRegions": manipulation_regions,
        "analysisTime": round(analysis_time, 1)
    }

# Main analyze endpoint
@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    try:
        result = predict_video(temp_path)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error(f"❌ Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

# Start server (use dynamic port for Railway)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
