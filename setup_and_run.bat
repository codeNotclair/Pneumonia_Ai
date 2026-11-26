@echo off
setlocal
cd /d "%~dp0"

echo Chest X-ray Classifier - local setup and launch

echo.
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python virtual environment...
    python -m venv .venv
    if errorlevel 1 goto setup_failed

    echo Installing pip updates...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed

    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo Installing PyTorch for CPU...
        ".venv\Scripts\python.exe" -m pip install torch torchvision
    ) else (
        echo NVIDIA driver detected. Installing CUDA-enabled PyTorch...
        ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    )
    if errorlevel 1 goto setup_failed

    echo Installing the web app and evaluation packages...
    ".venv\Scripts\python.exe" -m pip install streamlit pillow scikit-learn
    if errorlevel 1 goto setup_failed
)

if not exist "data\raw\archive\chest_xray\train\NORMAL" (
    echo.
    echo Dataset not found at data\raw\archive\chest_xray\train.
    echo Download the Kaggle dataset and extract it so train and test folders are under:
    echo data\raw\archive\chest_xray\
    echo Then double-click this file again.
    pause
    exit /b 1
)

if not exist "models\pneumonia_resnet18.pth" (
    echo No trained model found. Training now; the first run downloads pretrained weights.
    ".venv\Scripts\python.exe" train.py
    if errorlevel 1 goto setup_failed
)

echo Starting the app on this computer only...
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
exit /b %errorlevel%

:setup_failed
echo.
echo Setup or training failed. Review the error above, fix it, and run this file again.
pause
exit /b 1
