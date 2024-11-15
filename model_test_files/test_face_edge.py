import torch
from torchvision import transforms, models
from PIL import Image

# Load Face Edge model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('Models/Model1.pth', map_location=device))
model.eval()

# Define Face Edge transform
face_edge_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load and preprocess an image
image_path = "1.jpg"  # Replace with an actual test image
image = Image.open(image_path).convert("RGB")
image = face_edge_transform(image).unsqueeze(0).to(device)

# Run prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_names = ['Real', 'Fake']
    print("Prediction:", class_names[predicted.item()])
