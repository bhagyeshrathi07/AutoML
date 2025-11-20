import os
import json # Make sure this is imported
from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import run_automl

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Allow larger files (100MB) so bank-full.csv doesn't crash upload
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    target = request.form.get('target')
    
    # NEW: Get the list of models from the frontend
    models_json = request.form.get('models')
    selected_models = None
    if models_json:
        try:
            selected_models = json.loads(models_json)
        except:
            selected_models = None # Fallback to all models if JSON fails

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Pass the selected_models list to your pipeline
        results = run_automl(filepath, target, selected_models=selected_models)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)