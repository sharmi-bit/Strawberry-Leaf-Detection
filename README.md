Strawberry Plant Disease Detection

An AI-powered web app that detects diseases in strawberry plants from leaf images. It combines a U-Net segmentation model (to isolate the leaf from the background) with an XGBoost classifier (to identify the disease), served through a FastAPI backend with an HTML/CSS/JS frontend.

This project was trained on real data from the PlantVillage and Kaggle strawberry disease datasets, and was presented at the ICCCSS international conference.

How it works
Upload a photo of a strawberry leaf through the web interface.
Segmentation — a U-Net model isolates the leaf region from the background, so the classifier only looks at relevant plant tissue.
Classification — the segmented leaf is passed to an XGBoost model, which predicts the disease class (or healthy).
Result — the predicted disease and confidence score are shown in the browser.
Tech stack
Backend: FastAPI (Python)
Frontend: HTML, CSS, JavaScript
Models:
U-Net (PyTorch) — leaf segmentation
XGBoost — disease classification
Datasets: PlantVillage, Kaggle strawberry disease datasets
Project structure

Fill in / adjust this section to match your actual folder layout.

strawberry-disease-app/
  app/
    main.py                  # FastAPI entry point
    models/
      unet_segmentation.pth  # or .pt - U-Net weights
      xgboost_classifier.pkl # trained XGBoost model
    static/                  # CSS, JS
    templates/               # HTML pages
  requirements.txt
  README.md
Setup
Clone the repository
bash
   git clone <your-repo-url>
   cd strawberry-disease-app
Create a virtual environment
bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
Install dependencies
bash
   pip install -r requirements.txt
Run the app
bash
   uvicorn app.main:app --reload
Open http://localhost:8000 in your browser and upload a leaf image.
Model details
Stage	Model	Purpose
Segmentation	U-Net	Isolates the leaf from the background before classification
Classification	XGBoost	Predicts the disease class from the segmented leaf

Add your specific disease classes, training accuracy, and dataset size here once finalized (e.g. "Trained on X images across Y disease classes, achieving Z% test accuracy").

Known issues
Windows Smart App Control: on some Windows machines, Smart App Control blocks torch_python.dll from loading (WinError 4551), which prevents PyTorch from importing. If you hit this, either disable Smart App Control temporarily or run the app inside WSL/a Linux environment.
Future improvements
Expand the dataset to cover more disease classes and edge cases.
Add batch upload support for multiple leaf images at once.
Deploy the app so it's accessible without running it locally.

This README was drafted with placeholders for setup details (exact file paths, dependency versions, model file names, dataset size/accuracy) — fill these in with your actual project specifics before publishing.
