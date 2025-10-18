from flask import Flask, render_template, request, send_file, session, jsonify
import cv2
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EDITED_FOLDER'] = 'edited'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EDITED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        session['current_image'] = unique_filename
        session['original_image'] = unique_filename
        return jsonify({'success': True, 'filename': unique_filename})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/edit', methods=['POST'])
def edit_image():
    if 'current_image' not in session:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    action = request.json.get('action')
    value = request.json.get('value', None)
    current_file = session['current_image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'success': False, 'error': 'Error reading image'})

    # --- Image edits ---
    if action == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif action == 'blur':
        img = cv2.GaussianBlur(img, (15, 15), 0)
    elif action == 'rotate_left':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif action == 'rotate_right':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif action == 'flip_horizontal':
        img = cv2.flip(img, 1)
    elif action == 'flip_vertical':
        img = cv2.flip(img, 0)
    elif action == 'edge_detect':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif action == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
    elif action == 'brightness':
        val = int(value) if value else 30
        img = cv2.convertScaleAbs(img, alpha=1, beta=val)
    elif action == 'contrast':
        val = float(value) if value else 1.5
        img = cv2.convertScaleAbs(img, alpha=val, beta=0)
    elif action == 'sepia':
        kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        img = cv2.transform(img, kernel)
        img = np.clip(img,0,255).astype(np.uint8)
    elif action == 'vintage':
        kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        img = cv2.transform(img, kernel)
        img = np.clip(img,0,255).astype(np.uint8)
        rows, cols = img.shape[:2]
        X = cv2.getGaussianKernel(cols, cols/2)
        Y = cv2.getGaussianKernel(rows, rows/2)
        mask = (Y*X.T)/np.max(Y*X.T)
        mask = np.stack([mask]*3, axis=2)
        img = (img*mask).astype(np.uint8)
    elif action == 'cool':
        img = img.astype(np.float32)
        img[:,:,0] = np.clip(img[:,:,0]*1.2,0,255)
        img[:,:,1] = np.clip(img[:,:,1]*1.05,0,255)
        img[:,:,2] = np.clip(img[:,:,2]*0.9,0,255)
        img = img.astype(np.uint8)
    elif action == 'warm':
        img = img.astype(np.float32)
        img[:,:,0] = np.clip(img[:,:,0]*0.9,0,255)
        img[:,:,1] = np.clip(img[:,:,1]*1.05,0,255)
        img[:,:,2] = np.clip(img[:,:,2]*1.2,0,255)
        img = img.astype(np.uint8)
    elif action == 'negative':
        img = cv2.bitwise_not(img)
    elif action == 'emboss':
        kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        img = cv2.filter2D(img, -1, kernel)
    elif action == 'pencil_sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray,(21,21),0)
        sketch = cv2.divide(gray,255 - blur, scale=256)
        img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif action == 'cartoon':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        color = cv2.bilateralFilter(img,9,300,300)
        img = cv2.bitwise_and(color,color,mask=edges)

    edited_filename = f"edited_{uuid.uuid4()}.png"
    edited_path = os.path.join(app.config['EDITED_FOLDER'], edited_filename)
    cv2.imwrite(edited_path, img)

    session['current_image'] = edited_filename
    session['edited_path'] = edited_path
    return jsonify({'success': True, 'filename': edited_filename})

@app.route('/preview/<filename>')
def preview(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    edited_path = os.path.join(app.config['EDITED_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_file(upload_path)
    elif os.path.exists(edited_path):
        return send_file(edited_path)
    return 'File not found',404

@app.route('/download')
def download():
    if 'edited_path' in session:
        return send_file(session['edited_path'], as_attachment=True, download_name='edited_image.png')
    elif 'current_image' in session:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['current_image'])
        return send_file(filepath, as_attachment=True)
    return 'No image to download',400

@app.route('/reset')
def reset():
    if 'original_image' in session:
        session['current_image'] = session['original_image']
        return jsonify({'success': True, 'filename': session['original_image']})
    return jsonify({'success': False, 'error': 'No original image'})

if __name__ == '__main__':
    app.run(debug=True)