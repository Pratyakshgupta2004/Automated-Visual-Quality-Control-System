from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import base64
import io
import cv2
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = r"C:\avqc_project\runs\pcb_defect_v1\weights\best.pt"
CONFIDENCE = 0.25
# ──────────────────────────────────────────────────────────

app = Flask(__name__)

print("🧠 AI Model load ho raha hai...")
model = YOLO(MODEL_PATH)
print("✅ Model ready!")

# Colors for each defect
COLORS = {
    "missing_hole"    : (255, 0, 0),
    "mouse_bite"      : (255, 165, 0),
    "open_circuit"    : (255, 255, 0),
    "short"           : (0, 0, 255),
    "spur"            : (255, 0, 255),
    "spurious_copper" : (0, 255, 0),
}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PCB Defect Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #0f0f1a;
            color: white;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #00d4ff;
        }
        .header h1 { color: #00d4ff; font-size: 28px; }
        .header p  { color: #888; margin-top: 5px; }
        .container { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
        .upload-box {
            background: #1a1a2e;
            border: 2px dashed #00d4ff;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        .upload-box:hover { background: #16213e; border-color: #fff; }
        .upload-box h2 { color: #00d4ff; margin-bottom: 10px; }
        .upload-box p  { color: #888; }
        #fileInput { display: none; }
        .btn {
            background: linear-gradient(135deg, #00d4ff, #0080ff);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s;
        }
        .btn:hover { transform: scale(1.05); }
        .results { display: none; }
        .images-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .img-card {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 15px;
            text-align: center;
        }
        .img-card h3 { color: #00d4ff; margin-bottom: 10px; }
        .img-card img { width: 100%; border-radius: 10px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        .stat-card .number { font-size: 36px; font-weight: bold; }
        .stat-card .label  { color: #888; margin-top: 5px; }
        .ok     .number { color: #00ff88; }
        .defect .number { color: #ff4444; }
        .total  .number { color: #00d4ff; }
        .defect-list {
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .defect-list h3 { color: #00d4ff; margin-bottom: 15px; }
        .defect-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .defect-name { font-weight: bold; text-transform: capitalize; }
        .defect-conf {
            background: #ff4444;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 14px;
        }
        .status-ok     { color: #00ff88; font-size: 20px; font-weight: bold; }
        .status-defect { color: #ff4444; font-size: 20px; font-weight: bold; }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            color: #00d4ff;
            font-size: 18px;
        }
        .preview-img {
            max-width: 300px;
            max-height: 300px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 PCB Defect Detection System</h1>
        <p>AI-Powered Automated Visual Quality Control</p>
    </div>

    <div class="container">
        <!-- Upload Box -->
        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            <h2>📁 Image Upload Karo</h2>
            <p>PCB ki image yahan drop karo ya click karo</p>
            <img id="previewImg" class="preview-img" src="" alt="Preview">
            <br>
            <button class="btn" onclick="event.stopPropagation();" 
                    onmousedown="document.getElementById('fileInput').click()">
                📷 Image Choose Karo
            </button>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
        </div>

        <div style="text-align:center; margin-bottom: 30px;">
            <button class="btn" onclick="detectDefects()" style="font-size:18px; padding: 15px 50px;">
                🚀 Defect Detect Karo!
            </button>
        </div>

        <!-- Loading -->
        <div class="loading" id="loading">
            ⏳ AI analysis kar raha hai...
        </div>

        <!-- Results -->
        <div class="results" id="results">

            <!-- Status -->
            <div style="text-align:center; margin-bottom:20px;" id="statusText"></div>

            <!-- Images -->
            <div class="images-row">
                <div class="img-card">
                    <h3>📷 Original Image</h3>
                    <img id="originalImg" src="" alt="Original">
                </div>
                <div class="img-card">
                    <h3>🔍 Detected Defects</h3>
                    <img id="resultImg" src="" alt="Result">
                </div>
            </div>

            <!-- Stats -->
            <div class="stats">
                <div class="stat-card total">
                    <div class="number" id="totalDefects">0</div>
                    <div class="label">Total Defects</div>
                </div>
                <div class="stat-card ok">
                    <div class="number" id="statusOK">-</div>
                    <div class="label">Status</div>
                </div>
                <div class="stat-card defect">
                    <div class="number" id="confidence">-</div>
                    <div class="label">Max Confidence</div>
                </div>
            </div>

            <!-- Defect List -->
            <div class="defect-list">
                <h3>📋 Defect Details</h3>
                <div id="defectItems"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        function previewImage(event) {
            selectedFile = event.target.files[0];
            if (!selectedFile) return;
            const reader = new FileReader();
            reader.onload = e => {
                const img = document.getElementById('previewImg');
                img.src = e.target.result;
                img.style.display = 'block';
            };
            reader.readAsDataURL(selectedFile);
        }

        async function detectDefects() {
            if (!selectedFile) {
                alert('Pehle image choose karo!');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'block';

                // Images
                document.getElementById('originalImg').src = 'data:image/jpeg;base64,' + data.original;
                document.getElementById('resultImg').src   = 'data:image/jpeg;base64,' + data.result;

                // Stats
                document.getElementById('totalDefects').textContent = data.total_defects;

                const statusEl = document.getElementById('statusOK');
                const statusTx = document.getElementById('statusText');
                if (data.total_defects === 0) {
                    statusEl.textContent = '✅ OK';
                    statusEl.style.color = '#00ff88';
                    statusTx.innerHTML = '<span class="status-ok">✅ Product OK — Koi Defect Nahi!</span>';
                } else {
                    statusEl.textContent = '❌ FAIL';
                    statusEl.style.color = '#ff4444';
                    statusTx.innerHTML = '<span class="status-defect">❌ Defect Mila! Product Reject Karo!</span>';
                }

                // Confidence
                const maxConf = data.detections.length > 0
                    ? Math.max(...data.detections.map(d => d.confidence))
                    : 0;
                document.getElementById('confidence').textContent =
                    data.detections.length > 0 ? (maxConf * 100).toFixed(0) + '%' : 'N/A';

                // Defect list
                const itemsEl = document.getElementById('defectItems');
                if (data.detections.length === 0) {
                    itemsEl.innerHTML = '<p style="color:#00ff88; text-align:center;">✅ Koi defect nahi mila!</p>';
                } else {
                    itemsEl.innerHTML = data.detections.map(d => `
                        <div class="defect-item">
                            <span class="defect-name">⚠️ ${d.class_name.replace('_', ' ')}</span>
                            <span class="defect-conf">${(d.confidence * 100).toFixed(0)}% sure</span>
                        </div>
                    `).join('');
                }

            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + err.message);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Image nahi mili!'}), 400

    file  = request.files['image']
    img   = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Original image base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    original_b64 = base64.b64encode(buffer.getvalue()).decode()

    # AI Detection
    results = model(img_np, conf=CONFIDENCE, verbose=False)

    # Draw boxes
    result_img = img_np.copy()
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf     = float(box.conf[0])
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            color    = COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
            label = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_img, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(result_img, label, (x1+3, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            detections.append({
                'class_name': cls_name,
                'confidence': round(conf, 3),
                'bbox': [x1, y1, x2, y2]
            })

    # Result image base64
    result_pil = Image.fromarray(result_img)
    buffer2    = io.BytesIO()
    result_pil.save(buffer2, format='JPEG')
    result_b64 = base64.b64encode(buffer2.getvalue()).decode()

    return jsonify({
        'original'     : original_b64,
        'result'       : result_b64,
        'detections'   : detections,
        'total_defects': len(detections)
    })

if __name__ == '__main__':
    print("\n🌐 Dashboard chal raha hai!")
    print("   Browser mein kholo: http://localhost:5000")
    print("   Band karne ke liye: Ctrl+C\n")
    app.run(debug=False, host='0.0.0.0', port=5000)