import torch
import streamlit as st
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/pneumonia_resnet18.pth"

st.set_page_config(page_title="Chest X-ray Demo", page_icon="🩻")
st.title("Chest X-ray Classifier")
st.warning(
    "Educational prototype only. This model is not a medical device "
    "and must not be used to diagnose or guide treatment."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    classes = checkpoint["classes"]
    threshold = float(checkpoint.get("pneumonia_threshold", 0.5))

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes, threshold


model, classes, threshold = load_model()
normal_index = classes.index("NORMAL")
pneumonia_index = classes.index("PNEUMONIA")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = torch.softmax(model(image_tensor), dim=1)[0].cpu()

    predicted_index = (
        pneumonia_index if scores[pneumonia_index].item() >= threshold
        else normal_index
    )
    st.subheader(f"Model prediction: {classes[predicted_index]}")
    st.caption(
        f"PNEUMONIA decision threshold: {threshold:.2f} "
        "(selected using the validation split)."
    )
    st.write("Raw model scores (not calibrated medical probabilities):")
    for index, class_name in enumerate(classes):
        st.write(f"{class_name}: {scores[index].item():.1%}")
