import os
import json
import uuid
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from pipeline import run_automl
from codegen import generate_training_script

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Thread Executor (Prevents blocking)
executor = ThreadPoolExecutor(max_workers=2)
# In-memory storage for running tasks
tasks = {}

# --- CLEANUP ON STARTUP ---
# Delete old files so the server starts fresh
for f in glob.glob(os.path.join(MODELS_FOLDER, "*.pkl")):
    os.remove(f)
for f in glob.glob(os.path.join(UPLOAD_FOLDER, "*.csv")):
    os.remove(f)
print("🧹 Server Cleaned: Old temp files removed.")

# --- BACKGROUND WORKER ---
def background_task(task_id, filepath, target, selected_models):
    """Runs the pipeline in a separate thread and handles cleanup."""
    try:
        def update_progress(progress, message):
            # Callback to update global task state
            tasks[task_id]['progress'] = progress
            tasks[task_id]['logs'].append(message)

        # Run the heavy ML logic
        results = run_automl(filepath, target, selected_models, callback=update_progress)
        
        # Success State
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['results'] = results
        tasks[task_id]['logs'].append("✅ Pipeline completed successfully.")

    except Exception as e:
        # Failure State
        print(f"Task Failed: {e}")
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['logs'].append(f"❌ Error: {str(e)}")

    finally:
        # AUTO-DESTRUCT: Remove the user's CSV for privacy & space
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️ Deleted temp file: {filepath}")

# --- ROUTES ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    target = request.form.get('target')
    models_json = request.form.get('models')
    
    try:
        selected_models = json.loads(models_json) if models_json else None
    except:
        selected_models = None

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Initialize Task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "logs": ["🚀 Upload received. Initializing pipeline..."],
        "results": None
    }

    # Offload to background thread
    executor.submit(background_task, task_id, filepath, target, selected_models)

    return jsonify({"task_id": task_id, "message": "Processing started"})

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Frontend polls this endpoint to get progress bar updates"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route('/download-model', methods=['GET'])
def download_model():
    """Downloads the .pkl file"""
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({"error": "Model name required"}), 400

    filename = model_name.replace(" ", "_") + ".pkl"
    path = os.path.join(os.path.abspath(MODELS_FOLDER), filename)
    
    if not os.path.exists(path):
        return jsonify({"error": "Model file not found"}), 404

    return send_file(path, as_attachment=True)

@app.route('/download-code', methods=['GET'])
def download_code():
    """Generates a Python script to reproduce the model"""
    model_name = request.args.get('model')
    task_id = request.args.get('task_id')

    if not model_name or not task_id:
        return jsonify({"error": "Missing parameters"}), 400

    task = tasks.get(task_id)
    if not task or not task['results']:
        return jsonify({"error": "Task results not found"}), 404

    # Find the model configuration in the results
    model_data = next((m for m in task['results'] if m["Model"] == model_name), None)
    
    if not model_data:
        return jsonify({"error": "Model data not found"}), 404

    # Generate the code string
    script_content = generate_training_script(
        model_name,
        model_data.get("Best Params", {}),
        model_data.get("Task Type", "Classification"),
        "target_column"
    )

    return Response(
        script_content,
        mimetype="text/x-python",
        headers={"Content-disposition": f"attachment; filename=train_{model_name.replace(' ', '_')}.py"}
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)