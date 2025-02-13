from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from waitress import serve

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load the trained model
MODEL_PATH = "model/sign_language_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Update with actual dataset labels)
dataset_path = "dataset"  # Change this to your dataset path
CLASS_LABELS = {i: label for i, label in enumerate(sorted(os.listdir(dataset_path)))}

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((64, 64))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Validate file type (e.g., allow only images)
        allowed_extensions = {"png", "jpg", "jpeg", "gif"}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({"error": "Invalid file type. Only images are allowed."}), 400

        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_label = CLASS_LABELS.get(predicted_class, "Unknown")

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    # For development, use Flask's built-in server
    app.run(host="0.0.0.0", port=5000, debug=True)

    # For production, use Waitress
    # serve(app, host="0.0.0.0", port=5000)
