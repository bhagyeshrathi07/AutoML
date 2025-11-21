import os
import json
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pipeline import run_automl

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    target = request.form.get('target')
    
    models_json = request.form.get('models')
    selected_models = None
    if models_json:
        try:
            selected_models = json.loads(models_json)
        except:
            selected_models = None 

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        results = run_automl(filepath, target, selected_models=selected_models)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download', methods=['GET'])
def download_model():
    try:
        # Get the model name from the query parameter (e.g., ?model=XGBoost)
        model_name = request.args.get('model')
        
        if not model_name:
            return jsonify({"error": "Model name parameter required"}), 400

        # Convert "Logistic Regression" -> "Logistic_Regression.pkl"
        filename = model_name.replace(" ", "_") + ".pkl"
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, 'models', filename)
        
        if not os.path.exists(path):
             raise Exception(f"Model file '{filename}' not found.")

        return send_file(path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)