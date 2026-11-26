# Chest X-ray Classifier

A local, educational computer-vision demo that classifies an uploaded chest X-ray as **NORMAL** or **PNEUMONIA**. It uses transfer learning with a pretrained ResNet18 model, PyTorch, and Streamlit.

> **Important:** This is an experimental learning project, not a medical device. It must not be used to diagnose, rule out pneumonia, or guide treatment. Have medical images interpreted by a qualified clinician.

## What it does

- Trains a two-class image classifier on a Kaggle chest X-ray dataset.
- Evaluates the saved model on the dataset's test split.
- Runs a local browser interface for uploading a JPG or PNG and viewing the model's predicted class and raw class scores.
- Uses the NVIDIA GPU with CUDA when PyTorch detects one; otherwise it falls back to the CPU.

The web app binds to `127.0.0.1`, so it is intended to be accessed from the same computer. It is not deployed to a public domain.

## Project layout

```text
pneumonia-demo/
├── app.py                 # Local Streamlit upload app
├── train.py               # Stratified train/validation split and model training
├── evaluate.py            # Test split evaluation
├── data/                  # Local dataset files (ignored by Git)
├── models/                # Local trained weights (ignored by Git)
└── README.md
```

## Requirements

- Windows 10/11 and Python 3.10 or newer (Python 3.12 was used for development).
- Git, if you want to version the source code.
- An NVIDIA GPU is optional. CUDA-enabled PyTorch can use a compatible NVIDIA driver; the CUDA Toolkit does not need to be installed separately for the prebuilt PyTorch wheels.
- The first setup needs internet access to install packages and download ResNet18's pretrained weights. Training and app inference use the local files afterward.

## Setup

Open PowerShell in the project folder. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch. For an NVIDIA GPU, install the CUDA 12.8 wheel build:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For a CPU-only setup, use:

```powershell
python -m pip install torch torchvision
```

Install the remaining packages:

```powershell
python -m pip install streamlit pillow scikit-learn
```

To confirm whether PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset

Download the dataset from Kaggle and extract it so the folders look like this:

```text
data/raw/archive/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

`train.py` creates a stratified 85/15 train/validation split from `train/`. The dataset's supplied `val/` folder contains only a small number of images, so it is not used. `evaluate.py` uses `test/` for evaluation.

Check the dataset's license and terms before redistributing the data or any images. The `data/` and `models/` folders are ignored by Git; this repository contains source code, not the dataset or trained weights.

## Train and evaluate

From the project folder, with `.venv` active:

```powershell
python train.py
python evaluate.py
```

Training saves the best validation checkpoint to:

```text
models/pneumonia_resnet18.pth
```

The first training run downloads the pretrained ResNet18 weights. The training script runs three epochs and saves the checkpoint with the highest validation accuracy.

### Recorded test result

For the current training run, the confusion matrix was:

| Actual class | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| NORMAL | 117 | 117 |
| PNEUMONIA | 0 | 390 |

This corresponds to 81.2% overall accuracy, 50.0% recall for NORMAL, and 100.0% recall for PNEUMONIA on this dataset split. In particular, half of the NORMAL test images were incorrectly labeled PNEUMONIA. These results show that this model is not reliable for medical use. Dataset-level results do not establish performance on images from other hospitals, equipment, or patient populations.

The app's displayed scores are softmax outputs and are **not calibrated medical probabilities**.

## Run the local app

The trained checkpoint must exist first. Start Streamlit bound to loopback:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open the local URL printed in the terminal, usually [http://127.0.0.1:8501](http://127.0.0.1:8501). Stop the app with **Ctrl+C** in PowerShell.

## Git notes

The `.gitignore` excludes `.venv/`, `data/`, and `models/`, so large local files are not committed. A fresh clone needs the dataset and a newly trained model checkpoint before the app can run.
