from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import torch
from PIL import Image
import timm
import torch.nn as nn
import cv2
import os
from transformers import AutoModelForImageClassification, pipeline
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['Real', 'Fake']

# Load Xception model (Model2.pth) for face edge detection
def initialize_xception():
    model = timm.create_model('xception', pretrained=False)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 2)
    )
    return model.to(device)

# Load both Xception-based models
xception_model = initialize_xception()
xception_model.load_state_dict(torch.load('Models/Model2.pth', map_location=device))
xception_model.eval()

noise_model = initialize_xception()
noise_model.load_state_dict(torch.load('Models/best_model_epoch10_xcept.pth', map_location=device))
noise_model.eval()

# Load Vision Transformer (ViT) model and pipeline
vit_checkpoint = "Models/train_with_3666_vit/checkpoint-171"
vit_model = AutoModelForImageClassification.from_pretrained(vit_checkpoint)
vit_classifier = pipeline(
    task="image-classification", model=vit_model, image_processor=f"{vit_checkpoint}/preprocessor_config.json", device=0 if torch.cuda.is_available() else -1
)

# Define transformations for Xception model
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Temporary directory for processed images
temp_dir = 'temp_processed_images'
os.makedirs(temp_dir, exist_ok=True)

# Helper function to process images and enhance edges (consistent with training)
def enhance_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    return edges

# Preprocess function for face edge enhancement with saving as JPEG
def preprocess_edge_image(image_path):
    edges = enhance_edges(image_path)
    edge_image = Image.fromarray(edges)
    temp_image_path = os.path.join(temp_dir, f"edge_{os.path.basename(image_path)}")
    edge_image.save(temp_image_path, format='JPEG')
    return temp_image_path

# Preprocess function for noise model (Gaussian blur and noise extraction)
def preprocess_noise_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise = cv2.absdiff(img, blurred)
    noise_image = Image.fromarray(noise)
    temp_image_path = os.path.join(temp_dir, f"noise_{os.path.basename(image_path)}")
    noise_image.save(temp_image_path, format='JPEG')
    return temp_image_path

# Prediction function for the Vision Transformer model
def vit_predict(image_path):
    # Debugging statements to diagnose the prediction
    print("Running Vision Transformer prediction")
    
    # Run the prediction
    predictions = vit_classifier(image_path)
    print("Raw predictions:", predictions)  # Debugging: see raw output

    # Mapping model-specific labels to 'Real' or 'Fake'
    label_mapping = {"not_ai": "Real", "ai": "Fake"}
    
    # Find the best prediction based on score
    best_prediction = max(predictions, key=lambda x: x['score'])
    print("Best prediction:", best_prediction)  # Debugging: check chosen label

    # Map the label to the result using label_mapping
    result = label_mapping.get(best_prediction['label'], "Unknown")
    print("Mapped result:", result)  # Debugging: see final mapped result
    return result

# Prediction function for the Xception-based models
def predict_with_model(model, image_path, preprocess_fn, switch_labels=False):
    processed_image_path = preprocess_fn(image_path)
    image = Image.open(processed_image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    os.remove(processed_image_path)
    
    if switch_labels:
        predicted = 1 - predicted  # Switch 0 to 1 and 1 to 0
    
    return predicted.item()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data.get("image")
    model_type = data.get("model")

    if image_data:
        try:
            # Decode and save the image temporarily
            image_data = base64.b64decode(image_data.split(",")[1])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            temp_image_path = "temp_image.jpg"
            image.save(temp_image_path)

            # Select model based on user input
            if model_type == "face_edge":
                print("Using Xception model for face edge detection")
                predicted_class = predict_with_model(xception_model, temp_image_path, preprocess_edge_image, switch_labels=True)
                result = class_names[predicted_class]
            elif model_type == "noise":
                print("Using noise model")
                predicted_class = predict_with_model(noise_model, temp_image_path, preprocess_noise_image)
                result = class_names[predicted_class]
            elif model_type == "vit":
                print("Using Vision Transformer model")
                result = vit_predict(temp_image_path)
            else:
                return jsonify({"error": "Invalid model type"}), 400

            # Clean up
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            print(f"Prediction successful: {result}")
            return jsonify({"result": result})

        except Exception as e:
            print("Error during prediction:", str(e))
            return jsonify({"error": "An error occurred during prediction"}), 500

    return jsonify({"error": "No image data received"}), 400

if __name__ == '__main__':
    app.run(debug=True)
