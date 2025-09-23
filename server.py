from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://client-flora-fresh.vercel.app"]}})

# ================================
# CONFIG
# ================================
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# CLASS LABELS
# ================================
CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
# ================================
# LOAD MODEL
# ================================
student = models.resnet18(weights=None)
student.fc = nn.Linear(student.fc.in_features, NUM_CLASSES)
student.load_state_dict(torch.load("student_best.pth", map_location=DEVICE))
student = student.to(DEVICE)
student.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ================================
# ROUTE
# ================================

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = student(img)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = CLASS_NAMES[class_id]

    return jsonify({"prediction": class_name, "class_id": class_id})


# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run()
