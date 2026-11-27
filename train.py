from pathlib import Path
import re

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA_DIR = Path("data/raw/archive/chest_xray/train")
TEST_DIR = Path("data/raw/archive/chest_xray/test")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pneumonia_resnet18.pth"
BATCH_SIZE = 16
EPOCHS = 3
RANDOM_SEED = 42


def patient_group(filepath, class_name):
    """Return the patient prefix used by this Kaggle dataset's filenames."""
    stem = Path(filepath).stem
    if class_name == "NORMAL":
        match = re.match(r"(?:(?:NORMAL\d+)-)?(IM-\d+)", stem, re.IGNORECASE)
    elif class_name == "PNEUMONIA":
        match = re.match(r"(person\d+)", stem, re.IGNORECASE)
    else:
        match = None

    if match is None:
        raise ValueError(
            f"Cannot identify patient group for {class_name} image: {filepath}"
        )
    return f"{class_name}:{match.group(1).lower()}"


if not DATA_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {DATA_DIR}")
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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

# Patient IDs repeat across images. Exclude training images for any patient
# present in the official test split to prevent patient-level leakage.
train_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
eval_full = datasets.ImageFolder(DATA_DIR, transform=eval_transform)
test_data = datasets.ImageFolder(TEST_DIR, transform=eval_transform)
classes = train_full.classes
if test_data.classes != classes:
    raise ValueError(f"Train/test class mismatch: {classes} vs {test_data.classes}")
if set(classes) != {"NORMAL", "PNEUMONIA"}:
    raise ValueError(f"Expected NORMAL and PNEUMONIA folders; found {classes}")

train_groups = [
    patient_group(path, classes[label]) for path, label in train_full.samples
]
test_groups = {
    patient_group(path, test_data.classes[label]) for path, label in test_data.samples
}
eligible_ids = [
    index for index, group in enumerate(train_groups) if group not in test_groups
]
excluded_ids = [
    index for index, group in enumerate(train_groups) if group in test_groups
]
if not eligible_ids:
    raise ValueError("No training images remain after removing test-patient overlap")

eligible_labels = [train_full.samples[index][1] for index in eligible_ids]
eligible_groups = [train_groups[index] for index in eligible_ids]
splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_SEED
)
train_positions, val_positions = next(
    splitter.split(eligible_ids, eligible_labels, groups=eligible_groups)
)
train_ids = [eligible_ids[position] for position in train_positions]
val_ids = [eligible_ids[position] for position in val_positions]

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
print(f"Classes: {classes}")
print(
    f"Training images: {len(train_ids)} | Validation images: {len(val_ids)} "
    f"| Test-overlap training images excluded: {len(excluded_ids)}"
)
print("Validation is split by patient group; test-patient images are excluded from training.")

# Downloads pretrained weights on the first run if they are not cached.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

train_labels = torch.tensor([train_full.samples[index][1] for index in train_ids])
counts = torch.bincount(train_labels, minlength=len(classes)).float()
class_weights = len(train_ids) / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

normal_index = classes.index("NORMAL")
pneumonia_index = classes.index("PNEUMONIA")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_val_balanced_accuracy = -1.0

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
    val_actual, val_pneumonia_scores = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            val_pneumonia_scores.extend(
                probabilities[:, pneumonia_index].cpu().tolist()
            )
            val_actual.extend(targets.tolist())

    # Choose the decision threshold on validation only, maximizing balanced accuracy.
    best_threshold, epoch_balanced_accuracy = 0.5, -1.0
    for threshold_step in range(1, 100):
        threshold = threshold_step / 100
        val_predicted = [
            pneumonia_index if score >= threshold else normal_index
            for score in val_pneumonia_scores
        ]
        score = balanced_accuracy_score(val_actual, val_predicted)
        if score > epoch_balanced_accuracy or (
            score == epoch_balanced_accuracy
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = threshold
            epoch_balanced_accuracy = score

    val_predicted = [
        pneumonia_index if score >= best_threshold else normal_index
        for score in val_pneumonia_scores
    ]
    val_accuracy = accuracy_score(val_actual, val_predicted)
    average_loss = train_loss / len(train_ids)
    val_matrix = confusion_matrix(
        val_actual, val_predicted, labels=[normal_index, pneumonia_index]
    )
    print(
        f"Epoch {epoch + 1}/{EPOCHS} - loss: {average_loss:.4f} "
        f"- val accuracy: {val_accuracy:.3f} "
        f"- val balanced accuracy: {epoch_balanced_accuracy:.3f} "
        f"- pneumonia threshold: {best_threshold:.2f}"
    )
    print("Validation confusion matrix (rows=actual, columns=predicted):")
    print(val_matrix)

    if epoch_balanced_accuracy > best_val_balanced_accuracy:
        best_val_balanced_accuracy = epoch_balanced_accuracy
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "classes": classes,
                "pneumonia_threshold": best_threshold,
                "validation_balanced_accuracy": epoch_balanced_accuracy,
                "patient_group_split": True,
            },
            MODEL_PATH,
        )

print(f"Best validation balanced accuracy: {best_val_balanced_accuracy:.3f}")
print(f"Saved model and validation-selected threshold to: {MODEL_PATH}")
