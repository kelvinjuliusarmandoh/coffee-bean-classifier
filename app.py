import streamlit as st
from torchvision.transforms import Compose
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from typing import List, Tuple, Dict

# MODEL_PATH
TRAINED_MODEL_PATH = './models/trained-pretrained-model.pth'

# CLASS_NAMES
CLASS_NAMES = ['Dark', 'Green', 'Light', 'Medium']

#  Title for web app
st.title("Coffee-Bean-Classification:coffee:")
st.markdown("Welcome to Coffee-Bean Web App. Upload your image and get your prediction !")

@st.cache_data
def load_model(device):
    # Weights
    weights = EfficientNet_B1_Weights.DEFAULT

    # Transformer
    transformer = weights.transforms()

    # Instantiate the model
    effnet_model = efficientnet_b1(weights=weights)
    
    # Change classifier head
    effnet_model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=4)
    )

    # Load the weights that has been train
    trained_weights = torch.load("./models/trained-pretrained-model.pth", map_location=torch.device(device), weights_only=True)
    effnet_model.load_state_dict(trained_weights)
    return effnet_model, transformer

def load_and_pred(img_path, img_size, device=None):
    # Load model
    pretrained_model, transformer = load_model(device)

    # Read image
    image = Image.open(img_path)

    # Transform
    transformed_img = transformer(image)

    # To device
    pretrained_model.to(device)

    ## Predict on image ##
    pretrained_model.eval()
    with torch.inference_mode():
        # Unsqueeze
        transformed_img = transformed_img.unsqueeze(dim=0)

        # Make prediction
        logits = pretrained_model(transformed_img.to(device))

        # Probabilities
        probs = torch.softmax(logits, dim=1)

        # Label
        label = torch.argmax(probs, dim=1)

        # Pred class
        pred_class = CLASS_NAMES[label]

    return pred_class, probs

img_path = st.file_uploader(label="Upload your image...", type=['png','jpg'])
col1, col2 = st.columns(2)

if img_path:
    # Device agnostic 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the model and do prediction
    pred_class, probs = load_and_pred(img_path=img_path,
                                      img_size=(224, 224),
                                      device=device)

    df = pd.DataFrame({
        "Class": ["Dark", "Green", "Light", "Medium"],
        "Probability": [ round(probs[0][0].item(), 3),
                        round(probs[0][1].item(), 3),
                        round(probs[0][2].item(), 3),
                        round(probs[0][3].item(), 3)]
    })    
    with col1:
        st.header("Input Image")
        st.image(img_path) # Read image and show the image

    with col2:
        st.header("Output:")
        st.subheader("Prediction:")
        st.markdown(f"{pred_class}")
        st.subheader("Probabilities of Prediction:")
        st.write(df)
        