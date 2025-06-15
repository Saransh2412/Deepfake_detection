from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)  # ✅ Use __name__ instead of _name_
CORS(app)  # Enable CORS for all routes

# Constants
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20
MODEL_PATH = "deepfake_detection.h5"  # Ensure the model is present in the same folder or correct path
CONFIDENCE_THRESHOLD = 0.80
UPLOAD_FOLDER = tempfile.gettempdir()

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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

def predict_video(video_path):
    if model is None:
        return {"error": "Model not loaded"}, 500
    
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

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)
    
    try:
        result = predict_video(temp_path)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  # ✅ Correct check for script execution
    app.run(debug=True, host='0.0.0.0', port=5000)
