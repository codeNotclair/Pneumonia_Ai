\# Chest X-ray Classifier



A local educational demo that classifies uploaded chest X-rays as NORMAL or PNEUMONIA using a fine-tuned ResNet18 model.



\## Important limitation



This is not a medical device and must not be used for diagnosis or treatment. On this dataset's test split, the model correctly identified all 390 pneumonia images, but mislabeled 117 of 234 normal images as pneumonia (50% recall for NORMAL). Model scores are not calibrated medical probabilities.



\## Run locally



Activate the virtual environment, then start the app:



```powershell

.\\.venv\\Scripts\\Activate.ps1

python -m streamlit run app.py --server.address \_\_\_\_\_\_\_

