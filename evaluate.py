from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

TEST_DIR = Path("data/raw/archive/chest_xray/test")
MODEL_PATH = Path("models/pneumonia_resnet18.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
classes = checkpoint["classes"]

if test_data.classes != classes:
    raise ValueError(f"Class mismatch: model={classes}, test={test_data.classes}")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
actual, predicted = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predicted.extend(outputs.argmax(1).cpu().tolist())
        actual.extend(labels.tolist())

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(actual, predicted))
print("\nPer-class results:")
print(classification_report(actual, predicted, target_names=classes, digits=3))