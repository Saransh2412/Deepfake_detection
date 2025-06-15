from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import subprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
FRAMES_PER_VIDEO = 20
MODEL_PATH = "Deepfake_detection.h5"
GOOGLE_DRIVE_FILE_ID = "1pZiRmM-VXiSYc_SRI58zhGkCXkEnGB1K"
CONFIDENCE_THRESHOLD = 0.80

# ⬇️ Download the model using gdown
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            subprocess.run(
                ["gdown", f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", "-O", MODEL_PATH],
                check=True
            )
            print("Model downloaded.")
        except subprocess.CalledProcessError:
            print("❌ Failed to download model.")
            raise RuntimeError("Failed to download model from Google Drive.")

download_model()
model = load_model(MODEL_PATH, compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    frames = extract_frames(video_path)
    input_data = np.expand_dims(frames, axis=0)
    prediction = model.predict(input_data)[0][0]

    raw_label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if confidence < CONFIDENCE_THRESHOLD:
        label = "FAKE (Low Confidence)"
    else:
        label = raw_label

    return label, float(confidence)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type. Use .mp4'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        label, confidence = predict_video(filepath)
        os.remove(filepath)

        return jsonify({
            'status': 'success',
            'prediction': label,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
