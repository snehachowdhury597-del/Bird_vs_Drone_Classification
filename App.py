import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gdown
import os

## config
st.set_page_config(page_title="Bird vs Drone using EfficientNetB0", layout="centered")

# ----------------------------
# 1️⃣ Download model from Google Drive (if not exists)
# ----------------------------
file_id = "1MKL5cM83GjoyQ_lPCefqY9Z4EXB0ZgwI"
url = f"https://drive.google.com/uc?id={file_id}"
output = "model.pth"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# ----------------------------
# 2️⃣ Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights="DEFAULT")

    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1)
    )

    model.load_state_dict(
        torch.load("model.pth", map_location=torch.device("cpu"))
    )

    model.eval()
    return model

model = load_model()

# ----------------------------
# 3️⃣ Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------------------
# 4️⃣ UI
# ----------------------------
st.title("🐦🚁 Bird vs Drone Image Classification")
st.write("Upload an image of a bird or a drone and get prediction instantly!")

upload_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# 5️⃣ Prediction
# ----------------------------
if upload_file is not None:
    image = Image.open(upload_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).squeeze().item()

    if prob > 0.5:
        st.success(f"🚁 Drone detected with confidence: {prob:.2f}")
    else:
        st.success(f"🐦 Bird detected with confidence: {1-prob:.2f}")
