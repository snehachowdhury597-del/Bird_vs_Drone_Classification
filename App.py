import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

##config
st.set_page_config(page_title="Bird vs Drone using EfficientNetb0", layout="centered")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    #same classifier  training
    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256,1)
    )
    model.load_state_dict(torch.load('C:\\Users\\impor\\Downloads\\bird_vs_drone_app\\best_fine_tuned_efficientnet_freeze_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model
model=load_model()
##Transform
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
##UI page
st.title("Bird vs drone Image Classifiaction ")
st.write("Upload an image of a bird or a drone and the model will predict it for you😉")
upload_file=st.file_uploader("Upload an image of a Bird or Drone", type=["jpg","jpeg", "png"])

##prediction:
if upload_file is not None:
    image=Image.open(upload_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor=transform(image).unsqueeze(0) # Add batch dimension, shape: (1, 3, 224, 224)
    with torch.no_grad():
        output=model(input_tensor)
        prob=torch.sigmoid(output).squeeze().item()
    #results
    if prob>0.5:
        st.success(f"The model predicts that the image is a Drone with a confidence of {prob:.2f}")
    else:
        st.success(f"The model predicts that the image is a Bird with a confidence of {1-prob:.2f}")

