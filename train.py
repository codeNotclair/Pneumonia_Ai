from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA_DIR = Path("data/raw/archive/chest_xray/train")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pneumonia_resnet18.pth"
BATCH_SIZE = 16
EPOCHS = 3

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {DATA_DIR}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Use a stratified split from train; the supplied val folder is very small.
train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
eval_full = datasets.ImageFolder(DATA_DIR, transform=eval_transform)
labels = [label for _, label in train_full.samples]
train_ids, val_ids = train_test_split(
    list(range(len(labels))),
    test_size=0.15,
    random_state=42,
    stratify=labels,
)

train_loader = DataLoader(
    Subset(train_full, train_ids),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    Subset(eval_full, val_ids),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Classes: {train_full.classes}")
print(f"Training images: {len(train_ids)} | Validation images: {len(val_ids)}")

# Downloads pretrained weights the first time; internet is needed for that download.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_full.classes))
model = model.to(device)

counts = torch.bincount(torch.tensor(labels)[train_ids]).float()
class_weights = len(train_ids) / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images.to(device))
            correct += (outputs.argmax(1).cpu() == targets).sum().item()

    val_accuracy = correct / len(val_ids)
    average_loss = train_loss / len(train_ids)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- loss: {average_loss:.4f} "
        f"- validation accuracy: {val_accuracy:.3f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "classes": train_full.classes,
            },
            MODEL_PATH,
        )
        model.to(device)

print(f"Saved best model to: {MODEL_PATH}")